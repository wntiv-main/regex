"""Utilities for converting an NFA to a DFA, and optimising it"""

__author__ = "Callum Hynes"
__all__ = ["_OptimiseRegex"]

from abc import ABC, abstractmethod
from typing import Callable, Iterable, Self, override
import weakref
import numpy as np

# Pylint doesnt like loading sub-modules ig...?
# pylint: disable-next=no-name-in-module
from . import regex as rx  # Type annotating
from .regexutil import (ConsumeAny, MatchConditions, ParserPredicate,
                        SignedSet, State)


class _MovingIndexHandler(ABC):
    """
    Manages a set of indices attached to a iterable, so that concurrent
    modification and iteration are properly handled, by updating all
    references to indices within the list
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._internal_index})"


class _OptimiseRegex(_MovingIndexHandler):
    """
    Optimises a Regex to use minimal states without any epsilon moves or
    non-deterministic junctions
    """
    # Requires a lot of access to Regex objects ("friend" class)
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

    def _get_unreachable_at(
            self, state: State,
            *, ignore_paths_through: set[State] | None = None,
            terminate_at: set[State] | None = None) -> set[State]:
        """
        Determines if the given state is reachable from the terminate
        states, without passing through any of the given ignored states

        Arguments:
            state -- The state to check

        Keyword Arguments:
            ignore_paths_through -- States to ignore any paths through
                (default: {None})
            terminate_at -- States to terminate if reached
                (default: {None}: Terminate if start state is reached)

        Returns:
            A set of the unreachable states, empty if there are none
        """
        if terminate_at is None:
            terminate_at = {self.regex.start}
        todo: set[State] = {state}
        visited: set[State] = (set() if ignore_paths_through is None
                               else ignore_paths_through)
        first: bool = True
        while todo:
            s = todo.pop()
            if s in terminate_at:
                break
            if first:
                first = False
            # Must check at least first state
            elif s in visited:
                continue
            visited.add(s)
            todo |= set(np.argwhere(
                self.regex.edge_map[:, s]).flat)  # type: ignore
        else:  # No path to start
            return visited
        return set()

    def _remove_group_if_unreachable(self, start: State) -> bool:
        """
        Removes a given state if it deemed to be unreachable, other than
        through any loops or other weird self-referencial arrangements,
        by simply checking if a path can be traced back to the start
        state

        Arguments:
            start -- The state to check unreachability at (and remove,
                along with any unreachable peers)

        Returns:
            Whether the state was removed
        """
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
        # Lot of stuff here...  dont question it, it works
        # At some point in time i wrote this, knowing what i was doing
        # I no longer know what i was doing
        # see: programming meme now only god knows
        return (edge == MatchConditions.epsilon_transition
                or (edge in self.regex.edge_map[s1, s1]
                    and edge in self.regex.edge_map[s2, s2])
                or (edge in self.regex.edge_map[s1, s2]
                    and edge in self.regex.edge_map[s2, s1])
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
            diff = (self.regex.edge_map[i, s1]
                    ^ self.regex.edge_map[i, s2])
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
            diff = (self.regex.edge_map[s1, i]
                    ^ self.regex.edge_map[s2, i])
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
        # Ive decided that if you want a more detailed explenation of
        # What is happening here, you can read all of the DFA-related
        # wikipedia articles - they cover all the concepts involved here
        # including minification, epsilon closure, and powerset
        # construction.
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
                # Minimise twice for good measure???
                # (if it doesnt work i usually just copy/paste these
                #  lines and hope for the best ;) lol)
                self.minimise(i, j)
                if j.removed():
                    continue
                self.epsilon_closure(i, j)
                if j.removed():
                    continue
                if i.removed():
                    break
                self.minimise(i, j)
                if j.removed():
                    continue
                self.simple_powerset_construction(i, j)
            else:
                # > Powerset construction <
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
        num_inputs = self.regex._num_inputs(end.value(),
                                            exclude_self=True)
        num_outputs = self.regex._num_outputs(start.value(),
                                              exclude_self=True)
        start_loops = self.regex.edge_map[start.value(), start.value()]
        end_loops = self.regex.edge_map[end.value(), end.value()]
        if ((num_inputs == 1 or num_outputs == 1)
            and start_loops == end_loops):
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
            if self.regex._merge_outputs(start.value(), end.value()):
                self.todo.add(self.index(start))
            self.regex._debug(f"mrgd out {start} <- {end}")

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
            # Merge s2 self-loops also
            self.regex.connect_many(s1.value(), s1.value(),
                                    self.regex.edge_map[s2.value(),
                                                        s2.value()])
            self.regex._remove_state(s2.value())
            # Intended side-effect: will set s2's value to -1
            # Which will reset the caller's loop
            self.remove(s2)
            # Mark all input states as dirty
            for state, edges in enumerate(
                    self.regex.edge_map[:, s1.value()]):
                if edges:
                    self.todo.add(self.index(state))
            self.regex._debug(f"mergedi {s2} -> {s1}")
        elif self.can_minify_inputs(s1.value(), s2.value()):
            if s2.value() == self.regex.end:
                self.regex.end = s1.value()
            self.regex._merge_outputs(s1.value(), s2.value())
            # Merge s2 self-loops also
            self.regex.connect_many(s1.value(), s1.value(),
                                    self.regex.edge_map[s2.value(),
                                                        s2.value()])
            self.todo.add(self.index(s1))
            self.regex._remove_state(s2.value())
            # Intended side-effect: will set s2's value to -1
            # Which will reset the caller's loop
            self.remove(s2)
            self.regex._debug(f"merged {s2} -> {s1}")

    def simple_powerset_construction(
            self, start: _MovingIndex, end: _MovingIndex) -> None:
        """
        Merge multiple edges (between the same states) into one, if at
        all possible

        Arguments:
            start -- The starting state
            end -- The ending state
        """
        if len(self.regex.edge_map[start.value(), end.value()]) < 2:
            return # No two edges to merge
        accept: SignedSet[str] = SignedSet()
        to_remove: list[ParserPredicate] = []
        for edge in self.regex.edge_map[start.value(), end.value()]:
            match edge:
                case ConsumeAny():
                    accept |= edge.match_set
                    to_remove.append(edge)
                case _:
                    pass
        if accept.length() == 0:
            return # No moves
        if len(to_remove) < 2:
            return # No need to re-create
        for edge in to_remove:
            self.regex.edge_map[start.value(), end.value()].remove(edge)
        edge = ConsumeAny(accept)
        # Connect new merged edge
        self.regex.connect(start.value(), end.value(), edge)
        # pylint: disable-next=protected-access
        self.regex._debug(f"fixed {start} -> {end}")

    def _edges_remove_intersect(self,
                                edges: set[ParserPredicate],
                                intersect: SignedSet[str]) -> None:
        for edge in edges.copy():
            match edge:
                case ConsumeAny():
                    # Remove before mutate, re-add after if needed
                    edges.remove(edge)
                    edge.match_set -= intersect
                    if edge.match_set:
                        edges.add(edge)
                case MatchConditions.epsilon_transition:
                    pass
                case _:
                    raise NotImplementedError()

    def _edge_intersects(
        self,
        first: set[ParserPredicate],
        second: set[ParserPredicate]) -> set[ParserPredicate]:
        """
        Returns the intersecting edges of two connections, and removes
        those intersections from the connections. Note that the passed
        sets ARE mutated

        Arguments:
            first -- The first connection
            second -- The second connection

        Returns:
            Set of intersecting edges
        """
        coverage1 = SignedSet.union(
            *(x.coverage() for x in first
              if x != MatchConditions.epsilon_transition))
        coverage2 = SignedSet.union(
            *(x.coverage() for x in second
              if x != MatchConditions.epsilon_transition))
        intersection = coverage1 & coverage2
        if not intersection:
            return set() # No overlap, exit early
        # Overlap, need powerset
        intersect_edges: set[ParserPredicate] = set()
        # Remove intersection from both initial states
        for edges in first, second:
            for edge in edges.copy():
                if (edge in first and edge in second):
                    first.remove(edge)
                    second.remove(edge)
                    intersect_edges.add(edge.copy())
            self._edges_remove_intersect(edges, intersection)
        if (intersection - SignedSet.union(
            *(x.coverage() for x in intersect_edges
              if x != MatchConditions.epsilon_transition))):
            # If intersect not already covered, add ConsumeAny
            intersect_edges.add(ConsumeAny(intersection))
        return intersect_edges

    def powerset_construction(  # pylint: disable=design
            self, state: _MovingIndex,
            out1: _MovingIndex, out2: _MovingIndex) -> None:
        """
        Resolve any non-deterministic junctions from the given start
        state to the two given output states, in effect performing
        powerset construction, after repeated application over the graph
        (see: https://en.wikipedia.org/wiki/Powerset_construction). Note
        that here this is implemented by instead doing repeated
        "product"-set construction, with the same result

        Arguments:
            state -- The start state
            out1, out2 -- The output states
        """
        if out1.value() == out2.value():
            return
        # Check if sets have any overlap
        set1 = self.regex.edge_map[state.value(), out1.value()]
        set2 = self.regex.edge_map[state.value(), out2.value()]
        intersect_edges = self._edge_intersects(set1, set2)
        if not intersect_edges:
            return # No intersect to split off
        # States were changed, check again
        self.todo.add(self.index(out1))
        self.todo.add(self.index(out2))
        for out in out1, out2:
            other = out.value() ^ out1.value() ^ out2.value()
            # Special end state edge-case hope this works PLEASE
            if (out.value() == self.regex.end
                or MatchConditions.epsilon_transition
                    in self.regex.edge_map[out.value(),
                                           self.regex.end]):
                # Checks
                # i give up please stop asking me what this code does
                # how tf am i meant to know
                # it stops the forever loop bug thats all i know
                # As everyone knows, the "TEMPORARY, FIX SOON" solutions
                # are the ones that stick :D
                # Wait i think the fix im implementing now will make
                # this redundant (i hope cuz this is cursed)
                # I havent tested yet but maybe
                # Kinda think we still need it :/
                all_edges: Iterable[set[ParserPredicate]]\
                    = self.regex.edge_map[out.value(), :]
                end_out_coverage: SignedSet[str] = SignedSet.union(
                    *(x.coverage()
                      for edges in all_edges
                      for x in edges
                      if x != MatchConditions.epsilon_transition))
                # something to do with checking if nthe end state has an
                # output for all posssible chars, in which case powerset
                # construction can be problematic or smth idk i was not
                # thinking straight when i wrote this (nor am i now,
                # probably)
                if not end_out_coverage.negate():
                    self.regex.connect_many(state.value(),
                                    out.value(), intersect_edges)
                    self.regex._debug(f"endmrg {state} -> {out} <& "
                                      f"{other}")
                    return
            if (not self.regex.edge_map[state.value(), out.value()]
                and (self._get_unreachable_at(
                        out.value(),
                        ignore_paths_through={out.value()},
                        terminate_at={self.regex.start, other})
                and other != self.regex.end)):
                # Special case: One side covered by intersection
                self.regex.connect_many(state.value(),
                                   out.value(), intersect_edges)
                self.regex._merge_outputs(out.value(), other)
                intersect_loops = self.regex.edge_map[out.value(), other]
                loops = self.regex.edge_map[out.value(), out.value()]
                for edge in intersect_loops.copy():
                    if edge in loops:
                        intersect_loops.remove(edge)
                self.regex._debug(f"pstmrg {state} -> {out} <& {other}")
                if self._remove_group_if_unreachable(other):
                    self.regex._debug(f"pstmrg {state} -> {out} <& x")
                return
        # Add new state for the intersection
        new_state = self.index(self.regex.add_state())
        self.todo.add(self.index(new_state))
        self.regex.connect_many(state.value(), new_state.value(),
                                intersect_edges)
        # Intersection state should have outputs of both `out` states
        self.regex._merge_outputs(new_state.value(), out1.value())
        self.regex._merge_outputs(new_state.value(), out2.value())
        # Loops on both need moved to intersection state
        # This avoids infinite loops when there is a loop on both `out`
        # states
        loops1: set[ParserPredicate] = self.regex.edge_map[
            new_state.value(), out1.value()]
        loops2: set[ParserPredicate] = self.regex.edge_map[
            new_state.value(), out2.value()]
        self.regex.connect_many(new_state.value(), new_state.value(),
                                self._edge_intersects(loops1, loops2))
        main_loop_coverage = SignedSet.union(
            *(x.coverage() for x in self.regex.edge_map[
                new_state.value(), new_state.value()]
              if x != MatchConditions.epsilon_transition))
        self._edges_remove_intersect(loops1, main_loop_coverage)
        self._edges_remove_intersect(loops2, main_loop_coverage)
        msg = f"power2 {state} -> {out1} & {out2} -> {new_state}"
        for out in out1, out2:
            if out.value() == self.regex.end:
                # End should be immediately reachable, like it was
                # before the powerset construction
                self.regex.connect(new_state.value(), out.value(),
                                   MatchConditions.epsilon_transition)
                self.regex.end = out.value()
                self.todo.add(self.index(out))
            else:
                self._remove_group_if_unreachable(out.value())
        self.regex._debug(msg)
