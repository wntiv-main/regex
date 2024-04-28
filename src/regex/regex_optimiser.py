"""Utilities for converting an NFA to a DFA, and optimising it"""

__author__ = "Callum Hynes"
__all__ = ["_OptimiseRegex"]

from abc import ABC, abstractmethod
from typing import Callable, Iterable, Self, override
import weakref
import numpy as np

from .regex import Regex  # Type annotating
from .regexutil import (ConsumeAny, ConsumeString, MatchConditions,
                        ParserPredicate, SignedSet, State)


class _MovingIndexHandler(ABC):
    """
    Manages a set of indices attached to a iterable, so that concurrent
    modification and iteration are properly handled, by updating all
    references to indices withing the list
    """

    _instances: weakref.WeakSet['_MovingIndex']
    """Set of all indices referencing this list"""

    @abstractmethod
    def size(self):
        """The size of the list"""
        raise NotImplementedError()

    def __init__(self):
        # Use weak references so when indices are GCed, they can also be
        # removed from here. This allowed _MovingIndices to be treated
        # like any other object, and will self-destruct when they go out
        # of scope, automatically removing them from this list also.
        self._instances = weakref.WeakSet()

    def index(self, at: 'int | _MovingIndex') -> '_MovingIndex':
        """
        Create a handled reference to a list index

        Arguments:
            at -- The index to reference

        Returns:
            A reference to that index which will be updated accordingly
            when the list is modified
        """
        if isinstance(at, _MovingIndex):
            at = at.value()
        result = _MovingIndex(at)
        self._instances.add(result)
        return result

    def remove(self, index: 'int | _MovingIndex') -> None:
        """
        Mark an element removed from the list, and update the references
        into the list accordingly

        Arguments:
            index -- The index of the removed element
        """
        # pylint: disable=protected-access
        if isinstance(index, _MovingIndex):
            index = index.value()
        for inst in self._instances:
            if inst.value() > index:
                inst._internal_index -= 1
            elif inst.value() == index:
                inst._internal_index = -1

    def iterate(self, *,
                start: int = 0,
                end: '_MovingIndex | None' = None)\
            -> Iterable['_MovingIndex']:
        """
        Returns an iterator over the specified range. This iterator can
        safely be used while making concurrent modifications to the
        iterable, so long as handler.remove() is called appropriately.

        Yields:
            The current index as a `_MovingIndex`
        """
        condition: Callable[[], int]
        if end is None:
            condition = self.size
        else:
            condition = end.value
        i: _MovingIndex = self.index(start - 1)
        while i.next().value() < condition():
            yield i


class _MovingIndex:
    """
    An index into a list, updated by {_MovingIndexHandler} when the list
    is modified
    """

    _internal_index: int
    """The actual index into the list"""

    def __init__(self, at: State):
        self._internal_index = at

    def value(self) -> int:
        """
        Get the integral index into the list for access

        Returns:
            The internal index
        """
        return self._internal_index

    def next(self) -> Self:
        """
        Move to the next element in the list. Useful for iteration

        Returns:
            The current instance
        """
        self._internal_index += 1
        return self

    def removed(self) -> bool:
        """
        Returns:
            Whether this element has been removed
        """
        return self._internal_index == -1

    def reset_iteration(self) -> None:
        """Restart the iteration from the start of the list"""
        self._internal_index = -1

    def __str__(self) -> str:
        """
        Readable string representation

        Returns:
            String representation of the index
        """
        return str(self._internal_index)


class _OptimiseRegex(_MovingIndexHandler):
    """
    Optimises a Regex to use minimal states without any epsilon moves or
    non-deterministic junctions
    """
    # Requires a lot of access to Regex objects
    # pylint: disable=protected-access

    regex: Regex
    """The Regex to optimise"""

    todo: set[_MovingIndex]
    """Set of states that should be visited"""

    @override
    def size(self) -> int:
        """Amount of states to iterate"""
        return self.regex.size

    def __init__(self, regex: Regex):
        """
        Optimises a Regex to use minimal states without any epsilon moves or
        non-deterministic junctions

        Arguments:
            regex -- The Regex to optimise. This object WILL be mutated
        """
        super().__init__()
        self.regex = regex
        self.todo = set(map(self.index, range(self.regex.size)))
        self.optimise()

    def _get_unreachable_at(self, state: State) -> set[State]:
        todo: set[State] = {state}
        visited: set[State] = set()
        while todo:
            s = todo.pop()
            if s == self.regex.start:
                break
            if s in visited:
                continue
            visited.add(s)
            todo |= set(np.argwhere(
                self.regex.edge_map[:, s]).flat)  # type: ignore
        else:  # No path to start
            return visited
        return set()

    def _remove_group_if_unreachable(self, start: State) -> bool:
        states = self._get_unreachable_at(start)
        # Iterate in reverse to avoid shifting indices
        states = sorted(states, reverse=True)
        for state in states:
            self.regex._remove_state(state)
            self.remove(state)
        return bool(states)

    def _can_minify_between(self, edge: ParserPredicate,
                            s1: State, s2: State):
        """
        Returns whether an edge between two states should block the
        states from mwerging

        Arguments:
            edge -- The edge to check
            s1 -- The state which the edge is leaving
            s2 -- The state which the edge is entering

        Returns:
            Whether the states are still able to merge
        """
        return (edge == MatchConditions.epsilon_transition
                or (edge.kind_of_in(self.regex.edge_map[s1, s1])
                    and edge.kind_of_in(self.regex.edge_map[s2, s2]))
                or (edge.kind_of_in(self.regex.edge_map[s1, s2])
                    and edge.kind_of_in(self.regex.edge_map[s2, s1]))
                or (edge in self.regex.edge_map[s1, s1]
                    and MatchConditions.epsilon_transition
                    in self.regex.edge_map[s2, s1])
                or (edge in self.regex.edge_map[s1, s1]
                    and MatchConditions.epsilon_transition
                    in self.regex.edge_map[s1, s2]
                    and self.regex._num_inputs(s2,
                                               exclude_self=True) == 1))

    def can_minify_inputs(self, s1: State, s2: State) -> bool:
        """
        Compares the inputs of two states

        Returns:
            Whether the two states' inputs are similar enough that the
            states can be merged
        """
        if s1 == s2 or self.regex.start in {s1, s2}:
            return False
        for i in range(self.regex.size):
            diff = ParserPredicate.set_mutable_symdiff(
                self.regex.edge_map[i, s1],
                self.regex.edge_map[i, s2])
            for edge in diff:
                if i not in {s1, s2}:
                    return False
                j = i ^ s1 ^ s2  # {s1, s2} that is NOT i
                if not self._can_minify_between(edge, i, j):
                    return False
        return True

    def can_minify_outputs(self, s1: State, s2: State) -> bool:
        """
        Compares the outputs of two states

        Returns:
            Whether the two states' outputs are similar enough that the
            states can be merged
        """
        if s1 == s2 or (
            self.regex.end in {s1, s2}
            and not MatchConditions.epsilon_transition in
            self.regex.edge_map[
                # NOT end state
                self.regex.end ^ s1 ^ s2,
                # end state
                self.regex.end]):
            return False
        for i in range(self.regex.size):
            diff = ParserPredicate.set_mutable_symdiff(
                self.regex.edge_map[s1, i],
                self.regex.edge_map[s2, i])
            for edge in diff:
                if i not in {s1, s2}:
                    return False
                j = i ^ s1 ^ s2  # {s1, s2} that is NOT i
                if not self._can_minify_between(edge, i, j):
                    return False
        return True

    def optimise(self):  # pylint: disable=too-many-branches
        """
        Iterate the entire multi-digraph, performing optimisations where
        applicable
        """
        # Use task queue to allow reiteration if a state is "dirtied"
        while self.todo:
            i = self.todo.pop()
            if i.removed():
                continue
            if not __debug__ and i.value() > self.size():
                continue
            # Remove redundant states
            if self._remove_group_if_unreachable(i.value()):
                continue
            # Iterate states inner loop
            for j in self.iterate():
                self.minimise(i, j)
                if j.removed():
                    continue
                self.epsilon_closure(i, j)
                if j.removed():
                    continue
                if i.removed():
                    break
                self.minimise(i, j)
            else:
                # > Powerset construction <
                # While loop as expect size to change
                # Iterate lower half of triangle:
                #   0 1 2 3 ->
                # 0 \       (j)
                # 1 * \
                # 2 * * \
                # 3 * * * \
                # |
                # V (k)
                # This means that any states added during the iteration
                # will still be covered entirely
                for j in self.iterate(start=1):
                    if not self.regex.edge_map[i.value(), j.value()]:
                        continue  # fast-path for no edges
                    for k in self.iterate(end=j):
                        if not self.regex.edge_map[i.value(), k.value()]:
                            continue  # fast-path for no edges
                        self.powerset_construction(i, j, k)
                        if i.removed():
                            break
                    else:
                        continue
                    break  # continue break from above

    def epsilon_closure(self, start: _MovingIndex, end: _MovingIndex) \
            -> None:
        """
        Resolve any epsilon transitions from the given start state to
        the given end state
        """
        # Resolve epsilon transitions
        if start.value() == end.value():  # self-epsilon loops
            self.regex.edge_map[start.value(), end.value()].discard(
                MatchConditions.epsilon_transition)
            return  # only case when start == end
        if (MatchConditions.epsilon_transition
                not in self.regex.edge_map[start.value(), end.value()]):
            return  # return early if no epsilon moves
        num_inputs = self.regex._num_inputs(end.value(), exclude_self=True)
        num_outputs = self.regex._num_outputs(start.value(), exclude_self=True)
        start_loops = self.regex.edge_map[start.value(), start.value()]
        end_loops = self.regex.edge_map[end.value(), end.value()]
        if ((num_inputs == 1 or num_outputs == 1)
            and not ParserPredicate.set_mutable_symdiff(
                start_loops, end_loops)):
            # Trivial case, can simply merge two states
            self.regex.edge_map[start.value(), end.value()].remove(
                MatchConditions.epsilon_transition)
            if self.regex.end == end.value():
                self.regex.end = start.value()
            self.regex._merge(start.value(), end.value())
            self.regex._remove_state(end.value())
            self.remove(end)
            self.regex._debug(f"ez-closed {start} <- {end}")
        elif end.value() != self.regex.end:
            self.regex.edge_map[start.value(), end.value()].remove(
                MatchConditions.epsilon_transition)
            # Ensure other has no self-epsilon-loops
            self.regex.edge_map[end.value(), end.value()].discard(
                MatchConditions.epsilon_transition)
            self.regex._merge_outputs(start.value(), end.value())
            self.todo.add(self.index(end))
            self._remove_group_if_unreachable(end.value())
            self.regex._debug(f"e-closed {start} <- {end}")
            # Reset outer loop to give other states a chance to run
            self.todo.add(self.index(start))
            start.reset_iteration()
        else:
            # Merge outputs in the hope that these states can be merged
            # pass
            if self.regex._merge_outputs(start.value(), end.value()):
                self.todo.add(self.index(start))
            self.regex._debug(f"mrgd out {start} <- {end}")
        # self.regex.edge_map[start, end].remove(
        #     MatchConditions.epsilon_transition)
        # self.regex._merge_inputs(end, start)
        # for state in self.regex.edge_map[:, start].nonzero()[0]:
        #     self.todo.add(state[()])
        # self.todo.add(end)
        # if self.regex._remove_if_unreachable(start):
        #     self.shift_todo(start)
        #     self.regex._debug(f"e-closed inputs {end} <- {start}")
        #     return _ActionType.DELETED_START
        # self.regex._debug(f"e-closed inputs {end} <- {start}")

    def minimise(self, s1: _MovingIndex, s2: _MovingIndex) -> None:
        """
        Merge the two given states if possible, in order to minimise the
        amount of states in the resultant DFA
        """
        if self.can_minify_outputs(s1.value(), s2.value()):
            if s2.value() == self.regex.start:
                self.regex.start = s1.value()
            if s2.value() == self.regex.end:
                self.regex.end = s1.value()
            self.regex._merge_inputs(s1.value(), s2.value())
            self.regex._remove_state(s2.value())
            # Intended side-effect: will set s2's value to -1
            # Which will reset the caller's loop
            self.remove(s2)
            self.regex._debug(f"merged {s2} -> {s1}")
        elif self.can_minify_inputs(s1.value(), s2.value()):
            if s2.value() == self.regex.end:
                self.regex.end = s1.value()
            self.regex._merge_outputs(s1.value(), s2.value())
            self.regex._remove_state(s2.value())
            self.todo.add(self.index(s1))
            # Intended side-effect: will set s2's value to -1
            # Which will reset the caller's loop
            self.remove(s2)
            self.regex._debug(f"merged {s2} -> {s1}")

    def powerset_construction(  # pylint: disable=design
            self, state: _MovingIndex,
            out1: _MovingIndex, out2: _MovingIndex) -> None:
        """
        Resolve any non-deterministic junctions from the given start
        state to the two given output states

        Arguments:
            state -- The start state
            out1, out2 -- The output states
        """
        # Check if sets have any overlap
        set1 = self.regex.edge_map[state.value(), out1.value()]
        set2 = self.regex.edge_map[state.value(), out2.value()]
        # if MatchConditions.epsilon_transition in (row_set | column_set):
        #     if (out2.value() != self.regex.end
        #             and out1.value() != self.regex.end):
        #         # Unless e-move to end, retry
        #         self.todo.add(self.index(state))
        #         return
        coverage1 = SignedSet.union(
            *(x.coverage() for x in set1
              if x != MatchConditions.epsilon_transition))
        coverage2 = SignedSet.union(
            *(x.coverage() for x in set2
              if x != MatchConditions.epsilon_transition))
        intersection = coverage1 & coverage2
        if not intersection:
            return  # No overlap, exit early
        # Overlap, need powerset
        # Remove intersection from both initial states
        for edges in set1, set2:
            for edge in edges.copy():
                match edge:
                    case ConsumeAny():
                        edge.match_set -= intersection
                        if not edge.match_set:
                            edges.discard(edge)
                    case ConsumeString():
                        if edge.match_string in intersection:
                            edges.remove(edge)
                    case MatchConditions.epsilon_transition:
                        pass
                    case _:
                        raise NotImplementedError()
        # States were changed, check again
        self.todo.add(self.index(out1))
        self.todo.add(self.index(out2))
        intersect: ParserPredicate
        if intersection.length() == 1:
            intersect = ConsumeString(intersection.unwrap_value())
        else:
            intersect = ConsumeAny(intersection)
        for out in out1, out2:
            other = out.value() ^ out1.value() ^ out2.value()
            if out.value() == self.regex.end:
                #  Connecting to end is special-case
                # This is cursed just leave it be ;)
                outs = self.regex.edge_map[out.value(), other].copy()
                self.regex._merge_outputs(out.value(), other)
                self.regex._debug(f"endmrg {state} -> {out} <& {other}")
                for out_state in self.iterate():
                    if out_state.value() == out.value():
                        continue
                    if not self.regex.edge_map[
                            other, out_state.value()]:
                        continue  # Fast path
                    for out_end_state in self.iterate():
                        if out_end_state.value() == out.value():
                            continue # no self-loops for now
                        if not self.regex.edge_map[
                                out.value(), out_end_state.value()]:
                            continue # Fast path
                        if out_state.value() == out_end_state.value():
                            continue # Same `to` state
                        self.powerset_construction(
                            out, out_state, out_end_state)
                self.regex.edge_map[out.value(), other] = outs
                self.regex.connect(state.value(),
                                   out.value(), intersect)
                self.regex._debug(f"fndmrg {state} -> {out} <& {other}")
                return
            if (not self.regex.edge_map[state.value(), out.value()]
                and (self.regex._num_inputs(out.value(),
                                            exclude_self=True) == 0
                     or state.value() == self.regex.start)):
                # One side covered by intersection
                # other = out.value() ^ out1.value() ^ out2.value()
                self.regex.connect(state.value(),
                                   out.value(), intersect)
                self.regex._merge_outputs(out.value(), other)
                old_loops = self.regex.edge_map[out.value(), other]
                loops = self.regex.edge_map[out.value(), out.value()]
                for edge in old_loops.copy():
                    if edge.kind_of_in(loops):
                        old_loops.remove(edge)
                self._remove_group_if_unreachable(other)
                self.regex._debug(f"pstmrg {state} -> {out} <& {other}")
                return
        # Add new state for the intersection
        new_state = self.index(self.regex.add_state())
        self.todo.add(self.index(new_state))
        self.regex.connect(state.value(), new_state.value(), intersect)
        # Connect outputs
        # if (not self.regex.edge_map[state.value(), out1.value()]
        #         and not self.regex.edge_map[state.value(), out2.value()]):
        #     # Loop, use e-moves to connect
        #     self.regex.connect(new_state.value(), out1.value(),
        #                        MatchConditions.epsilon_transition)
        #     self.regex.connect(new_state.value(), out2.value(),
        #                        MatchConditions.epsilon_transition)
        #     self.regex._debug(f"power {state} -> {out1} & {out2} -> "
        #                       f"{new_state}")
        #     loops1: set[ParserPredicate] = self.regex.edge_map[
        #         out1.value(), out1.value()]
        #     loops2: set[ParserPredicate] = self.regex.edge_map[
        #         out2.value(), out2.value()]
        #     for edge in loops1.copy():
        #         if edge.kind_of_in(loops2) is not None:
        #             # Shared loops
        #             self.regex.connect(new_state.value(), new_state.value(),
        #                                edge.copy())
        # if self.regex.edge_map[out2.value(), state.value()]:
        #     out1, out2 = out2, out1
        # if self.regex._remove_if_unreachable(out1.value()):
        #     self.remove(out1)
        # else:
        #     self.epsilon_closure(new_state, out1)
        # if self.regex._remove_if_unreachable(out2.value()):
        #     self.remove(out2)
        # elif not new_state.removed():
        #     self.epsilon_closure(new_state, out2)
        # return
        # Otherwise, merge
        self.regex._merge_outputs(new_state.value(), out1.value())
        self.regex._merge_outputs(new_state.value(), out2.value())
        loops1: set[ParserPredicate] = self.regex.edge_map[
            new_state.value(), out1.value()]
        loops2: set[ParserPredicate] = self.regex.edge_map[
            new_state.value(), out2.value()]
        old_loops: set[ParserPredicate] = self.regex.edge_map[
            new_state.value(), new_state.value()]
        for edge in loops1.copy():
            if (other := edge.kind_of_in(loops2)) is not None:
                # Shared loops
                self.regex.connect(new_state.value(), new_state.value(),
                                   edge.copy())
        for edge in old_loops:
            if (other := edge.kind_of_in(loops1)):
                loops1.remove(other)
            if (other := edge.kind_of_in(loops2)):
                loops2.remove(other)
        msg = f"power2 {state} -> {out1} & {out2} -> {new_state}"
        for out in out1, out2:
            if out.value() == self.regex.end:
                self.regex.connect(out.value(), new_state.value(),
                                   MatchConditions.epsilon_transition)
                self.regex.end = out.value()
                self.todo.add(self.index(out))
            else:
                self._remove_group_if_unreachable(out.value())
        self.regex._debug(msg)
