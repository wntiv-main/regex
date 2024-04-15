__all__ = ["Regex", "State"]
__author__ = "Callum Hynes"

from typing import Any, Callable, Self, Sequence, TypeAlias, overload
try:
    import numpy as np
except ImportError:
    print("Library requires numpy: `pip install numpy`")
    exit()

from regex_factory import _RegexFactory
from regexutil import MatchConditions, ParserPredicate, State

class Regex:
    # Function for debugging
    _debug_function: Callable[['Regex', str], None]\
        = lambda *_: None

    def _debug(self, msg: str):
        Regex._debug_function(self, msg)

    # S_n = x where table[S_(n-1), x].any(x=>x(ctx) is True)
    edge_map: np.ndarray[set[ParserPredicate]]
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
                result.edge_map = Regex._empty_arr((2, 2))
                result.start = 0
                result.end = 1
                result.connect(result.start, result.end, x)
                return result
            case (Regex() as x,), {}:
                result = super().__new__(cls)
                # Deep copy sets within table
                result.edge_map = np.vectorize(
                    Regex._inner_copy_set)(x.edge_map)
                result.start = x.start
                result.end = x.end
                return result
            case (), {"_privated": _}:
                result = super().__new__(cls)
                result.edge_map = Regex._empty_arr((1, 1))
                result.start = 0
                result.end = 0
                return result
            case _:
                raise TypeError(f"Invalid args to {cls.__name__}(): ",
                                args, kwargs)

    @property
    def size(self) -> int:
        assert (len(self.edge_map.shape) == 2
                and self.edge_map.shape[0]
                == self.edge_map.shape[1])
        return self.edge_map.shape[0]
    __len__ = size.fget

    @staticmethod
    def _empty_arr(size: Sequence[int]):
        # if only np.fromfunction() worked :(
        return np.vectorize(lambda _: set())(np.empty(size, dtype=set))

    def add_state(self) -> State:
        # Resize table to have new state at end
        self._diagonal_block_with(Regex._empty_arr((1, 1)))
        return self.size - 1

    def connect(self,
                start_state: State,
                end_state: State,
                connection: ParserPredicate) -> None:
        if not self.edge_map[start_state, end_state]:
            self.edge_map[start_state, end_state] = set()
        self.edge_map[start_state, end_state].add(connection)

    def connect_many(self,
                     start_state: State,
                     end_state: State,
                     connections: set[ParserPredicate]) -> None:
        if not self.edge_map[start_state, end_state]:
            self.edge_map[start_state, end_state] = set()
        # Deep copy needed
        for edge in connections:
            self.edge_map[start_state, end_state].add(edge.copy())

    def _num_inputs(self, state: State) -> int:
        return (np.sum(np.vectorize(len)(self.edge_map[:, state]))
                # Phantom input to start state
                + (state == self.start))

    def _num_outputs(self, state: State) -> int:
        return (np.sum(np.vectorize(len)(self.edge_map[state, :]))
                # Phantom ouput from end state
                + (state == self.end))

    def _remove_if_unreachable(self, state: State) -> bool:
        if ((not self.edge_map[:, state].any()
                and not state == self.start)
            or (not self.edge_map[state, :].any()
                and not state == self.end)):
            self._remove_state(state)
            return True
        return False

    def _remove_state(self, state: State) -> None:
        if self.start > state:
            self.start -= 1
        if self.end > state:
            self.end -= 1
        # remove both row and column for `state`
        self.edge_map = np.delete(
            np.delete(
                self.edge_map,
                state, 0),
            state, 1)

    def _merge_outputs(self, s1_idx: State, s2_idx: State) -> None:
        # Iterate s2 row and make same connections from s1
        it = np.nditer(self.edge_map[s2_idx, :],
                       flags=['c_index', 'refs_ok'])
        for edges in it:
            # thank numpy for [()]
            self.connect_many(s1_idx, it.index, edges[()])

    def _merge_inputs(self, s1_idx: State, s2_idx: State) -> None:
        # Iterate s2 column and make same connections to s1
        it = np.nditer(self.edge_map[:, s2_idx],
                       flags=['c_index', 'refs_ok'])
        for edges in it:
            # thank numpy for [()]
            self.connect_many(it.index, s1_idx, edges[()])

    def _merge(self, s1_idx: State, s2_idx: State) -> None:
        self._merge_inputs(s1_idx, s2_idx)
        self._merge_outputs(s1_idx, s2_idx)

    def _diagonal_block_with(self, other: np.ndarray):
        # constructs block matrix like:
        # [[self,  empty]
        #  [empty, other]]
        self.edge_map = np.block([
            [self.edge_map, Regex._empty_arr((self.size,
                                                      other.shape[1]))],
            [Regex._empty_arr((other.shape[0], self.size)), other]])

    # TODO: non-inline versions
    def __iadd__(self, other: Any) -> Self:
        if isinstance(other, Regex):
            other: Regex = other.copy()
            offset = self.size
            self._diagonal_block_with(other.edge_map)
            # Connect our end to their start
            self.connect(self.end, offset + other.start,
                         MatchConditions.epsilon_transition)
            self.end = offset + other.end
        elif isinstance(other, ParserPredicate):
            new_state = self.add_state()
            self.connect(self.end, new_state, other)
            self.end = new_state
        else:
            raise NotImplementedError()
        return self

    def __add__(self, other: Any) -> 'Regex':
        result = self.copy()
        result += other
        return result

    def __imul__(self, scalar: int) -> Self:
        for _ in range(scalar):
            self += self

    def __mul__(self, scalar: int) -> Self:
        result = self.copy()
        result *= scalar
        return result

    def __ior__(self, other: 'Regex') -> Self:
        other = other.copy()
        offset = self.size
        self._diagonal_block_with(other.edge_map)
        # Connect our start to their start
        self.connect(self.start, offset + other.start,
                     MatchConditions.epsilon_transition)
        # Connect our end to their end
        self.connect(self.end, offset + other.end,
                     MatchConditions.epsilon_transition)
        self.end = offset + other.end
        return self

    def __or__(self, other: 'Regex') -> 'Regex':
        result = self.copy()
        result |= other
        return result

    def is_in(self, input: str) -> bool:
        ctx = MatchConditions(input)
        state = self.start
        while state != self.end:
            for i in range(self.size):
                for edge in self.edge_map[state, i]:
                    edge: ParserPredicate
                    if edge.evaluate(ctx):
                        state = i
                        break  # break to outer loop
                else:
                    continue
                break
            else:
                # No match
                return False
        return True

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
            for row in self.edge_map]), self.start, self.end)
