"""Utilities for converting an NFA to a DFA, and optimising it"""

__author__ = "Callum Hynes"
__all__ = ["_OptimiseRegex"]

from abc import ABC, abstractmethod
from typing import Callable, Iterable, Self, override
import weakref

import regex as rx  # Type annotating
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
    # Required a lot of access to Regex objects
    # pylint: disable=protected-access

    regex: 'rx.Regex'
    """The Regex to optimise"""

    todo: set[_MovingIndex]
    """Set of states that should be visited"""

    @override
    def size(self) -> int:
        """Amount of states to iterate"""
        return self.regex.size

    def __init__(self, regex: 'rx.Regex'):
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
            if i in {s1, s2}:
                diff = ParserPredicate.set_mutable_symdiff(
                    self.regex.edge_map[i, s1],
                    self.regex.edge_map[i, s2])
                for edge in diff:
                    if edge != MatchConditions.epsilon_transition:
                        return False
            elif (self.regex.edge_map[i, s1]
                  != self.regex.edge_map[i, s2]):
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
            if i in {s1, s2}:
                diff = ParserPredicate.set_mutable_symdiff(
                    self.regex.edge_map[s1, i],
                    self.regex.edge_map[s2, i])
                for edge in diff:
                    if edge != MatchConditions.epsilon_transition:
                        return False
            elif (self.regex.edge_map[s1, i]
                  != self.regex.edge_map[s2, i]):
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
            # Remove redundant states
            if self.regex._remove_if_unreachable(i.value()):
                self.remove(i)
                continue
            # Iterate states inner loop
            for j in self.iterate():
                # TODO: soon edges will have more info
                self.epsilon_closure(i, j)
                if j.removed():
                    continue
                if i.removed():
                    break
                # minimisation
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

        if (self.regex._num_inputs(end.value()) == 1
                or self.regex._num_outputs(start.value()) == 1):
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
            if self.regex._remove_if_unreachable(end.value()):
                self.remove(end)
            else:
                self.todo.add(self.index(end))
            self.regex._debug(f"e-closed {start} <- {end}")
            # Reset outer loop to give other states a chance to run
            self.todo.add(self.index(start))
            start.reset_iteration()
        else:
            # Merge outputs in the hope that these states can be merged
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

    def powerset_construction(
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
        row_set = self.regex.edge_map[state.value(), out1.value()]
        column_set = self.regex.edge_map[state.value(), out2.value()]
        # if MatchConditions.epsilon_transition in (row_set | column_set):
        #     if (out2.value() != self.regex.end
        #             and out1.value() != self.regex.end):
        #         # Unless e-move to end, retry
        #         self.todo.add(self.index(state))
        #         return
        row_coverage = SignedSet.union(
            *(x.coverage() for x in row_set
              if x != MatchConditions.epsilon_transition))
        column_coverage = SignedSet.union(
            *(x.coverage() for x in column_set
              if x != MatchConditions.epsilon_transition))
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
                case MatchConditions.epsilon_transition:
                    pass
                case _:
                    raise NotImplementedError()
        # States were changed, check again
        self.todo.add(self.index(out1))
        self.todo.add(self.index(out2))
        # Add new state for the intersection
        new_state = self.index(self.regex.add_state())
        self.todo.add(self.index(new_state))
        # TODO: assuming that intersect should be ConsumeAny
        intersect: ParserPredicate
        if intersection.length() == 1:
            intersect = ConsumeString(intersection.unwrap_value())
        else:
            intersect = ConsumeAny(intersection)
        self.regex.connect(state.value(), new_state.value(), intersect)
        # Connect outputs
        self.regex.connect(new_state.value(), out1.value(),
                           MatchConditions.epsilon_transition)
        self.regex.connect(new_state.value(), out2.value(),
                           MatchConditions.epsilon_transition)
        self.regex._debug(f"power {state} -> {out1} & {out2} -> "
                          f"{new_state}")
        if self.regex._remove_if_unreachable(out1.value()):
            self.remove(out1)
        else:
            self.epsilon_closure(new_state, out1)
        if self.regex._remove_if_unreachable(out2.value()):
            self.remove(out2)
        elif not new_state.removed():
            self.epsilon_closure(new_state, out2)
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
