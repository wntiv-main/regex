

from enum import IntEnum, IntFlag, auto
import regex as rx
from regexutil import ConsumeAny, ConsumeString, MatchConditions, ParserPredicate, SignedSet, State


class _ActionType(IntFlag):
    NONE = 0
    DELETED_START = auto()
    DELETED_END = auto()
    MERGED_TO_START = auto()
    MERGED_END_TO_START = DELETED_END | MERGED_TO_START


# snake-case name as functional-like interface
class _optimise_regex:
    regex: 'rx.Regex'
    todo: set[State]

    def __init__(self, regex: 'rx.Regex') -> None:
        self.regex = regex
        self.todo = set(range(self.regex.size))
        self.optimise()

    def shift_todo(self, after: State):
        # Fix indices in list after removing state
        temp = set()
        for i in self.todo:
            if i > after:
                temp.add(i - 1)
            elif i != after:
                temp.add(i)
        self.todo = temp

    def can_minify_inputs(self, s1: State, s2: State) -> bool:
        if s1 == s2 or s1 == self.regex.start or s2 == self.regex.start:
            return False
        for i in range(self.regex.size):
            if i == s1 or i == s2:
                diff = (self.regex.edge_map[i, s1]
                        ^ self.regex.edge_map[i, s2])
                for edge in diff:
                    if edge != MatchConditions.epsilon_transition:
                        return False
            elif (self.regex.edge_map[i, s1]
                  != self.regex.edge_map[i, s2]):
                return False
        return True

    def can_minify_outputs(self, s1: State, s2: State) -> bool:
        if s1 == s2 or s1 == self.regex.end or s2 == self.regex.end:
            return False
        for i in range(self.regex.size):
            if i == s1 or i == s2:
                diff = (self.regex.edge_map[s1, i]
                        ^ self.regex.edge_map[s2, i])
                for edge in diff:
                    if edge != MatchConditions.epsilon_transition:
                        return False
            elif (self.regex.edge_map[s1, i]
                  != self.regex.edge_map[s2, i]):
                return False
        return True

    def optimise(self):
        # Use task queue to allow reiteration if a state is "dirtied"
        while self.todo:
            i = self.todo.pop()
            # Remove redundant states
            if self.regex._remove_if_unreachable(i):
                self.shift_todo(i)
                continue
            # Iterate states inner loop
            j = 0
            while j < self.regex.size:
                # TODO: soon edges will have more info
                result = self.epsilon_closure(i, j)
                if result & _ActionType.DELETED_END:
                    if i > j:
                        i -= 1
                if result & _ActionType.MERGED_TO_START:
                    j = 0
                    continue
                if result & _ActionType.DELETED_START:
                    break
                # minimisation
                if self.minimise(i, j):
                    if i > j:
                        i -= 1
                    j = 0
                    continue
                j += 1
            else:
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
                while j < self.regex.size:
                    k = 0
                    while k < j:
                        deleted = self.powerset_construction(i, j, k)
                        # Adjust iteration indices if states were deleted
                        for state in deleted:
                            if i == state:
                                break  # break all way to outer loop
                            if i > state:
                                i -= 1
                            if j >= state:
                                j -= 1
                            if k >= state:
                                k -= 1
                        else:
                            k += 1
                            continue
                        break  # continue break from above
                    else:
                        j += 1
                        continue
                    break  # continue break from above

    def epsilon_closure(self, start: State, end: State) -> _ActionType:
        # Resolve epsilon transitions
        if start == end:  # self-epsilon loops
            self.regex.edge_map[start, end].discard(
                MatchConditions.epsilon_transition)
            return _ActionType.NONE  # only case when start == end
        if (MatchConditions.epsilon_transition
                not in self.regex.edge_map[start, end]):
            return _ActionType.NONE  # return early if no epsilon moves

        if (self.regex._num_inputs(end) == 1
                or self.regex._num_outputs(start) == 1):
            # Trivial case, can simply merge two states
            self.regex.edge_map[start, end].remove(
                MatchConditions.epsilon_transition)
            if self.regex.end == end:
                self.regex.end = start
            self.regex._merge(start, end)
            self.regex._remove_state(end)
            self.shift_todo(end)
            self.regex._debug(f"ez-closed {start} <- {end}")
            return _ActionType.MERGED_END_TO_START

        if end != self.regex.end:
            self.regex.edge_map[start, end].remove(
                MatchConditions.epsilon_transition)
            self.regex._merge_outputs(start, end)
            self.todo.add(start)
            result = _ActionType.MERGED_TO_START
            if self.regex._remove_if_unreachable(end):
                self.shift_todo(end)
                result |= _ActionType.DELETED_END
            self.regex._debug(f"e-closed {start} <- {end}")
            return result
        # elif (self.regex._num_inputs(j) == 1
        #       or self.regex._num_outputs(i) == 1):
        #     self.regex.edge_map[i, j].remove(
        #         MatchConditions.epsilon_transition)
        #     self.regex._merge(j, i)
        #     self.regex._remove_state(i)
        #     shift_todo(i)
        #     self.regex._debug(f"e-closed inputs {j} <- {i}")
        #     # Deleted `i` state, goto next `i` state
        #     countinue_outer_loop = True
        #     break

        self.regex.edge_map[start, end].remove(
            MatchConditions.epsilon_transition)
        self.regex._merge_inputs(end, start)
        for state in self.regex.edge_map[:, start].nonzero()[0]:
            self.todo.add(state[()])
        self.todo.add(end)
        if self.regex._remove_if_unreachable(start):
            self.shift_todo(start)
            self.regex._debug(f"e-closed inputs {end} <- {start}")
            return _ActionType.DELETED_START
        self.regex._debug(f"e-closed inputs {end} <- {start}")
        return _ActionType.NONE

    def minimise(self, s1: State, s2: State) -> bool:
        if self.can_minify_outputs(s1, s2):
            if s2 == self.regex.start:
                self.regex.start = s1
            self.regex._merge_inputs(s1, s2)
            self.regex._remove_state(s2)
            # State removed, handle shifted indices
            self.shift_todo(s2)
            self.regex._debug(f"merged {s2} -> {s1}")
            return True
        if self.can_minify_inputs(s1, s2):
            if s2 == self.regex.end:
                self.regex.end = s1
            self.regex._merge_outputs(s1, s2)
            self.regex._remove_state(s2)
            self.todo.add(s1)
            # State removed, handle shifted indices
            self.shift_todo(s2)
            self.regex._debug(f"merged {s2} -> {s1}")
            return True
        return False

    def powerset_construction(
            self, state: State,
            out1: State, out2: State) -> list[State]:
        # Check if sets have any overlap
        row_set = self.regex.edge_map[state, out1]
        column_set = self.regex.edge_map[state, out2]
        if MatchConditions.epsilon_transition in (row_set | column_set):
            self.todo.add(state)
            return []
        row_coverage = SignedSet.union(
            *map(lambda x: x.coverage(), row_set))
        column_coverage = SignedSet.union(
            *map(lambda x: x.coverage(), column_set))
        intersection = row_coverage & column_coverage
        if not intersection:
            return []  # No overlap, exit early
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
        self.todo.add(out1)
        self.todo.add(out2)
        # Add new state for the intersection
        new_state = self.regex.add_state()
        self.todo.add(new_state)
        # TODO: assuming that intersect should be ConsumeAny
        intersect: ParserPredicate
        if intersection.length() == 1:
            intersect = ConsumeString(intersection.unwrap_value())
        else:
            intersect = ConsumeAny(intersection)
        self.regex.connect(state, new_state, intersect)
        # Connect outputs
        self.regex.connect(new_state, out1,
                           MatchConditions.epsilon_transition)
        self.regex.connect(new_state, out2,
                           MatchConditions.epsilon_transition)
        self.regex._debug(f"power {state} -> {out1} & {out2} -> {new_state}")
        removed: list[State] = []
        if self.regex._remove_if_unreachable(out1):
            self.shift_todo(out1)
            removed.append(out1)
            if out2 > out1:
                out2 -= 1
            if new_state > out1:
                new_state -= 1
        else:
            res1 = self.epsilon_closure(new_state, out1)
            if res1 & _ActionType.DELETED_END:
                removed.append(out1)
                if out2 > out1:
                    out2 -= 1
                if new_state > out1:
                    new_state -= 1
            if res1 & _ActionType.DELETED_START:
                removed.append(new_state)
                if out2 > new_state:
                    out2 -= 1
        if self.regex._remove_if_unreachable(out2):
            self.shift_todo(out2)
            removed.append(out2)
        elif new_state not in removed:
            res2 = self.epsilon_closure(new_state, out2)
            if res2 & _ActionType.DELETED_END:
                removed.append(out2)
            if res2 & _ActionType.DELETED_START:
                removed.append(new_state)
        # with np.nditer(
        #         [self.regex.edge_map[out1, :],
        #          self.regex.edge_map[out2, :],
        #          self.regex.edge_map[new_state, :]],
        #         flags=['refs_ok'],
        #         op_flags=[['readonly'], ['readonly'], ['writeonly']]) as it:
        #     for i1, i2, o in it:
        #         o[...] = i1 | i2

        # for j in range(self.regex.size):
        #     for edge in self.regex.edge_map[out1, j]\
        #             | self.regex.edge_map[out2, j]:
        #         self.regex.connect(new_state, j, edge.copy())
        return removed
