__all__ = ["Regex"]
__author__ = "Callum Hynes"

from typing import Any, Callable, Self, Sequence, TypeAlias, overload
import numpy as np

from regex_factory import _RegexFactory
from regexutil import MatchConditions, ParserPredicate

State: TypeAlias = int

class Regex:
    # Function for debugging
    _debug_function: Callable[['Regex', str], None] = lambda *_: None

    def _debug(self, msg: str):
        Regex._debug_function(self, msg)

    # S_n = x where table[S_(n-1), x].any(x=>x(ctx) is True)
    transition_table: np.ndarray[set[ParserPredicate]]
    start: State
    end: State

    @overload
    def __new__(cls, regex: str) -> Self:
        ...

    @overload
    def __new__(cls, *, _privated: None) -> Self:
        ...

    @overload
    def __new__(
            cls,
            other: 'Regex') -> Self:
        ...

    @overload
    def __new__(
            cls,
            predicate: ParserPredicate,
            *, _privated: None) -> Self:
        ...

    def __new__(cls, *args, **kwargs) -> Self:
        match args, kwargs:
            case (str(regex),), {}:
                return _RegexFactory(regex).build()
            case (ParserPredicate() as x,), {"_privated": _}:
                result = super().__new__(cls)
                result.transition_table = Regex._empty_arr((2, 2))
                result.start = 0
                result.end = 1
                result.connect(result.start, result.end, x)
                return result
            case (Regex() as x,), {}:
                result = super().__new__(cls)
                # Deep copy sets within table
                result.transition_table = np.vectorize(
                    Regex._inner_copy_set)(x.transition_table)
                result.start = x.start
                result.end = x.end
                return result
            case (), {"_privated": _}:
                result = super().__new__(cls)
                result.transition_table = Regex._empty_arr((1, 1))
                result.start = 0
                result.end = 0
                return result
            case _:
                raise TypeError(f"Invalid args to {cls.__name__}()", args)

    @property
    def size(self) -> int:
        assert (len(self.transition_table.shape) == 2
                and self.transition_table.shape[0]
                == self.transition_table.shape[1])
        return self.transition_table.shape[0]
    __len__ = size.fget

    @staticmethod
    def _empty_arr(size: Sequence[int]):
        # if only np.fromfunction() worked :(
        return np.vectorize(lambda _: set())(np.empty(size, dtype=set))

    def append_state(self) -> State:
        # Resize table to have new state at end
        self._diagonal_block_with(Regex._empty_arr((1, 1)))
        return self.size - 1

    def connect(self,
                start_state: State,
                end_state: State,
                connection: ParserPredicate) -> None:
        if not self.transition_table[start_state, end_state]:
            self.transition_table[start_state, end_state] = set()
        self.transition_table[start_state, end_state].add(connection)

    def connect_many(self,
                     start_state: State,
                     end_state: State,
                     connections: set[ParserPredicate]) -> None:
        if not self.transition_table[start_state, end_state]:
            self.transition_table[start_state, end_state] = set()
        self.transition_table[start_state, end_state] |= connections

    def _can_minify(self, s1: State, s2: State) -> bool:
        if s1 == s2:
            return False
        for i in range(self.size):
            if i == s1 or i == s2:
                diff = (self.transition_table[s1, i]
                        ^ self.transition_table[s2, i])
                for edge in diff:
                    if edge != MatchConditions.epsilon_transition:
                        return False
            elif (self.transition_table[s1, i]
                  != self.transition_table[s2, i]):
                return False
        return True

    def _epsilon_closure(self):
        to_remove: set[State] = set()
        for i in range(self.size):
            for j in range(self.size):
                if i in to_remove or j in to_remove:
                    continue
                # TODO: soon edges will have more info
                if (MatchConditions.epsilon_transition
                        in self.transition_table[i, j]
                        and j != self.end):
                    self._merge_outputs(i, j)
                    self.transition_table[i, j].remove(
                        MatchConditions.epsilon_transition)
                    self._debug(f"e-closed {i} -> {j}")
                # minimisation
                if self._can_minify(i, j):
                    self._merge(i, j)
                    to_remove.add(j)
        # Remove in reverse to avoid deletions mis-ordering the matrix
        for state in sorted(to_remove, reverse=True):
            self._remove_state(state)

    def _minimisation(self):
        to_remove: set[State] = set()
        for i in range(self.size - 1):
            for j in range(i + 1, self.size):
                # TODO: more robust comparison
                if self.transition_table[i] == self.transition_table[j]:
                    self._merge(i, j)
                    to_remove.add(j)
        # Remove in reverse to avoid deletions mis-ordering the matrix
        for state in sorted(to_remove, reverse=True):
            self._remove_state(state)

    def _remove_state(self, state: State) -> None:
        if self.start > state:
            self.start -= 1
        if self.end > state:
            self.end -= 1
        # remove both row and column for `state`
        self.transition_table = np.delete(
            np.delete(
                self.transition_table,
                state, 0),
            state, 1)

    def _merge_outputs(self, s1_idx: State, s2_idx: State) -> None:
        # Iterate s2 row and make same connections from s1
        it = np.nditer(self.transition_table[s2_idx, :],
                       flags=['c_index', 'refs_ok'])
        for edges in it:
            self.connect_many(s1_idx, it.index, edges)

    def _merge_inputs(self, s1_idx: State, s2_idx: State) -> None:
        # Iterate s2 column and make same connections to s1
        it = np.nditer(self.transition_table[:, s2_idx],
                       flags=['c_index', 'refs_ok'])
        for edges in it:
            self.connect_many(it.index, s1_idx, edges)

    def _merge(self, s1_idx: State, s2_idx: State) -> None:
        self._merge_inputs(s1_idx, s2_idx)
        self._merge_outputs(s1_idx, s2_idx)

    def _diagonal_block_with(self, other: np.ndarray):
        # constructs block matrix like:
        # [[self,  empty]
        #  [empty, other]]
        self.transition_table = np.block([
            [self.transition_table, Regex._empty_arr((self.size,
                                                      other.shape[1]))],
            [Regex._empty_arr((other.shape[0], self.size)), other]])

    def __iadd__(self, other: Any) -> Self:
        if isinstance(other, Regex):
            other = other.copy()
            offset = self.size
            self._diagonal_block_with(other.transition_table)
            # Connect our end to their start
            self.connect(self.end, offset + other.start,
                         MatchConditions.epsilon_transition)
            self.end = offset + other.end
        elif isinstance(other, ParserPredicate):
            new_state = self.append_state()
            self.connect(self.end, new_state, other)
            self.end = new_state
        else:
            raise NotImplementedError()
        return self

    def __ior__(self, other: 'Regex') -> Self:
        other = other.copy()
        offset = self.size
        self._diagonal_block_with(other.transition_table)
        # Connect our start to their start
        self.connect(self.start, offset + other.start,
                     MatchConditions.epsilon_transition)
        # Connect our end to their end
        self.connect(self.end, offset + other.end,
                     MatchConditions.epsilon_transition)
        self.end = offset + other.end
        return self

    def optional(self) -> Self:
        self.connect(self.start, self.end,
                     MatchConditions.epsilon_transition)
        return self

    def repeated(self) -> Self:
        self.connect(self.end, self.start,
                     MatchConditions.epsilon_transition)
        return self

    @staticmethod
    def _inner_copy_set(obj: Any):
        if isinstance(obj, set):
            return obj.copy()
        return None

    def copy(self):
        return Regex(self)

    def __str__(self) -> str:
        return "[%s]: %d -> %d" % (',\n '.join([
            "[%s]" % ', '.join([
                f"{{{', '.join([str(edge) for edge in edges])}}}"
                if isinstance(edges, set) else "{}"
                for edges in row])
            for row in self.transition_table]), self.start, self.end)
