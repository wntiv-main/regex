__all__ = ["Regex", "State"]
__author__ = "Callum Hynes"

from typing import Any, Callable, Self, Sequence, TypeAlias, overload
try:
    import numpy as np
except ImportError:
    print("Library requires numpy: `pip install numpy`")
    exit()

from regex_factory import _RegexFactory
from regexutil import ConsumeAny, ConsumeString, MatchConditions, ParserPredicate, SignedSet

State: TypeAlias = int

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
        self.edge_map[start_state, end_state] |= connections

    def _can_minify_outputs(self, s1: State, s2: State) -> bool:
        if s1 == s2 or s1 == self.end or s2 == self.end:
            return False
        for i in range(self.size):
            if i == s1 or i == s2:
                diff = (self.edge_map[s1, i]
                        ^ self.edge_map[s2, i])
                for edge in diff:
                    if edge != MatchConditions.epsilon_transition:
                        return False
            elif (self.edge_map[s1, i]
                  != self.edge_map[s2, i]):
                return False
        return True

    def _num_inputs(self, state: State) -> int:
        return np.sum(np.vectorize(len)(self.edge_map[:, state]))

    def _num_outputs(self, state: State) -> int:
        return np.sum(np.vectorize(len)(self.edge_map[state, :]))

    def _can_minify_inputs(self, s1: State, s2: State) -> bool:
        if s1 == s2 or s1 == self.start or s2 == self.start:
            return False
        for i in range(self.size):
            if i == s1 or i == s2:
                diff = (self.edge_map[i, s1]
                        ^ self.edge_map[i, s2])
                for edge in diff:
                    if edge != MatchConditions.epsilon_transition:
                        return False
            elif (self.edge_map[i, s1]
                  != self.edge_map[i, s2]):
                return False
        return True

    def _remove_if_unreachable(self, state: State) -> bool:
        if (not self.edge_map[:, state].any()
                and not state == self.start):
            self._remove_state(state)
            return True
        return False

    def _optimise(self):
        # Use task queue to allow reiteration if a state is "dirtied"
        todo: set[State] = set(range(self.size))

        def shift_todo(after: State):
            # Fix indices in list after removing state
            nonlocal todo
            temp = set()
            for i in todo:
                if i > after:
                    temp.add(i - 1)
                elif i != after:
                    temp.add(i)
            todo = temp

        while todo:
            i = todo.pop()
            # Remove redundant states
            if self._remove_if_unreachable(i):
                shift_todo(i)
                continue
            # Iterate states inner loop
            countinue_outer_loop: bool = False
            j = 0
            while j < self.size:
                # TODO: soon edges will have more info
                # Resolve epsilon transitions
                if (MatchConditions.epsilon_transition
                        in self.edge_map[i, j]):
                    if i == j:
                        self.edge_map[i, j].remove(
                            MatchConditions.epsilon_transition)
                        self._debug(f"e-closed loop on {i}")
                    elif j == self.end and (self._num_inputs(j) == 1
                                            or self._num_outputs(i) == 1):
                        self.edge_map[i, j].remove(
                            MatchConditions.epsilon_transition)
                        self._merge(j, i)
                        self._remove_state(i)
                        shift_todo(i)
                        self._debug(f"e-closed inputs {j} <- {i}")
                        # Deleted `i` state, goto next `i` state
                        countinue_outer_loop = True
                        break
                    elif j != self.end:
                        self._merge_outputs(i, j)
                        self.edge_map[i, j].remove(
                            MatchConditions.epsilon_transition)
                        todo.add(i)
                        if self._remove_if_unreachable(j):
                            shift_todo(j)
                            if i > j:
                                i -= 1
                        else:
                            todo.add(j)
                        self._debug(f"e-closed {i} <- {j}")
                        j = 0  # Merged outputs, reset loop
                        continue
                # minimisation
                if self._can_minify_outputs(i, j):
                    if j == self.start:
                        self.start = i
                    self._merge_inputs(i, j)
                    self._remove_state(j)
                    # State removed, handle shifted indices
                    shift_todo(j)
                    if i > j:
                        i -= 1
                    self._debug(f"merged {j} -> {i}")
                    j = 0  # Merged states, reset loop
                    continue
                if self._can_minify_inputs(i, j):
                    if j == self.end:
                        self.end = i
                    self._merge_outputs(i, j)
                    self._remove_state(j)
                    # State removed, handle shifted indices
                    shift_todo(j)
                    if i > j:
                        i -= 1
                    self._debug(f"merged {j} -> {i}")
                    todo.add(i)
                    j = 0  # Merged states, reset loop
                    continue
                j += 1
            if countinue_outer_loop:
                continue
            # > Powerset construction <
            # While loop as expect size to change
            # Iterate lower half of triangle:
            #   0 1 2 3
            # 0 \
            # 1 * \
            # 2 * * \
            # 3 * * * \
            # This means that any states added during the iteration will
            # still be covered entirely
            j = 1
            while j < self.size:
                k = 0
                while k < j:
                    self._powerset_construction(todo, i, j, k)
                    k += 1
                j += 1

    def _powerset_construction(
            self, todo: set[State], state: State,
            out1: State, out2: State):
        # Check if sets have any overlap
        row_set = self.edge_map[state, out1]
        column_set = self.edge_map[state, out2]
        if MatchConditions.epsilon_transition in (row_set | column_set):
            todo.add(state)
            return
        row_coverage = SignedSet.union(
            *map(lambda x: x.coverage(), row_set))
        column_coverage = SignedSet.union(
            *map(lambda x: x.coverage(), column_set))
        intersection = row_coverage & column_coverage
        if not intersection:
            return  # No overlap, exit early
        # Overlap, need powerset
        # Remove intersection from both initial states
        for edge in row_set | column_set:
            match edge:
                case ConsumeAny():
                    edge.match_set -= intersection
                    if not edge.match_set:
                        row_set.discard(edge)
                        column_set.discard(edge)
                case ConsumeString():
                    if edge.match_string in intersection:
                        row_set.discard(edge)
                        column_set.discard(edge)
                case _:
                    raise NotImplementedError()
        # States were changed, check again
        todo.add(out1)
        todo.add(out2)
        # Add new state for the intersection
        new_state = self.add_state()
        # TODO: assuming that intersect should be ConsumeAny
        self.connect(state, new_state, ConsumeAny(intersection))
        # Connect outputs
        self.connect(new_state, out1,
                     MatchConditions.epsilon_transition)
        self.connect(new_state, out2,
                     MatchConditions.epsilon_transition)
        # for j in range(self.size):
        #     for edge in self.transition_table[out1, j]\
        #             | self.transition_table[out2, j]:
        #         self.connect(new_state, j, edge)
        self._debug(f"power {state} -> {out1} & {out2} -> {new_state}")
        todo.add(new_state)

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
            self.connect_many(s1_idx, it.index, edges)

    def _merge_inputs(self, s1_idx: State, s2_idx: State) -> None:
        # Iterate s2 column and make same connections to s1
        it = np.nditer(self.edge_map[:, s2_idx],
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
