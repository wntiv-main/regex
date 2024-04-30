"""Main Regex class for representing regular expressions"""

__author__ = "Callum Hynes"
__all__ = ["Regex"]

from typing import (Any, Callable, Iterable, Mapping, Optional, Self, Sequence,
                    overload)

import imp
try:
    import numpy as np
except ImportError as e:
    e.add_note("Regex library requires numpy: `pip install numpy`")
    raise e  # raise to user

from .regexutil import MatchConditions, ParserPredicate, State


class Regex:
    """
    Represents a regular expression, using a deterministic finite
    automaton structure.
    """
    if __debug__:
        # Function for debugging
        _debug_function: Callable[['Regex', str], None]\
            = lambda *_: None
        """
        A function which gets called during certain important steps of
        the regex building process, in order to allow logging of certain
        events.

        Arguments:
            regex -- The Regex object, in it's current state.
            msg -- A debug message explaining what is happening.
        """

    def _debug(self, msg: str):
        """
        Call the debugging function.

        Arguments:
            msg -- Description of why the function was called, or
                what is happening at the current point in time.
        """
        if __debug__:
            Regex._debug_function(self, msg)

    _base: 'Regex | None' = None
    """The Regex that this was based off of"""

    _reverse: 'Regex | None' = None
    """The reverse of this Regex"""

    # pylint: disable-next=no-member, unsubscriptable-object
    edge_map: np.ndarray[Any, np.dtypes.ObjectDType]
    """
    A 2D array (or matrix) of sets of transitions, so that from a given
    input state q, the q'th row (edge_map[q, :]) is a list of sets of
    the transitions, indexed so that the output state is p if any of the
    transitions in edge_map[q, p] can match. As such, the rows represent
    the outputs from their respective states, and the columns represent
    the *inputs* to their respectived states.
    """

    start: State
    """The index of the start state"""

    end: State
    """The index of the end state"""

    @overload
    def __new__(cls, pattern: str) -> Self:
        """
        Creates a new Regex from the given regular expression pattern.

        Arguments:
            pattern -- the pattern to match, as a string
        """

    @overload
    # Require _privated arg to prevent accidental use
    def __new__(cls, *, _privated: None) -> Self:
        """Creates an empty regex object, only used internally"""

    @overload
    def __new__(
            cls,
            other: 'Regex') -> Self:
        """
        Create a Regex from another Regex, copying it.

        Arguments:
            other -- The regex to copy from.
        """

    @overload
    # Require _privated arg to prevent accidental use
    def __new__(
            cls,
            predicate: ParserPredicate,
            *, _privated: None) -> Self:
        """
        Create a Regex with a single transition from the start to end,
        used only internally.

        Arguments:
            predicate -- The transition connecting start to end
        """

    # Implementation of above
    def __new__(cls, *args, **kwargs) -> Self:
        # Use match statement to determine which overload
        match args, kwargs:
            case (str(regex),), {}:
                # Import here to resolve cyclic import
                # pylint: disable-next=import-outside-toplevel
                from .regex_factory import _RegexFactory
                return _RegexFactory(regex).build()  # type: ignore
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
                result.edge_map = np.vectorize(set.copy)(x.edge_map)
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
        """The total amount of states in the DFA"""
        # array should be 2D square
        assert (len(self.edge_map.shape) == 2
                and self.edge_map.shape[0]
                == self.edge_map.shape[1])
        return self.edge_map.shape[0]
    __len__ = size.fget

    @staticmethod
    def _empty_arr(size: Sequence[int]) \
            -> np.ndarray:
        """
        Create a ndarray of the given shape, filled with unique empty
        sets.

        Arguments:
            size -- A tuple representing the shape, size and
                dimensionality of the array to be created.

        Returns:
            A new numpy ndarray.
        """
        # if only np.fromfunction() worked as intended :(
        return np.vectorize(lambda _: set())(np.empty(size, dtype=set))

    def add_state(self) -> State:
        """
        Adds an extra state to the DFA.

        Returns:
            The index of the new state.
        """
        # Resize table to have new state at end
        self._diagonal_block_with(Regex._empty_arr((1, 1)))
        return self.size - 1

    def connect(self,
                start_state: State,
                end_state: State,
                connection: ParserPredicate) -> None:
        """
        Connects the two given states with the given transition.

        Arguments:
            start_state -- The state from which the transition should
                start.
            end_state -- The state at which the transition should end.
            connection -- The transition which should connect the two
                states.
        """
        # if not self.edge_map[start_state, end_state]:
        #     self.edge_map[start_state, end_state] = set()
        if not connection.kind_of_in(self.edge_map[start_state,
                                                   end_state]):
            self.edge_map[start_state, end_state].add(connection)

    def connect_many(self,
                     start_state: State,
                     end_state: State,
                     connections: Iterable[ParserPredicate]) -> None:
        """
        Connects all the given connections between the two given states.

        Arguments:
            start_state -- The state that all the transitions should
                start at.
            end_state -- The state that all the transitions should end
                at.
            connections -- A collection of all the transitions that
                should join the two states.
        """
        # if not self.edge_map[start_state, end_state]:
        #     self.edge_map[start_state, end_state] = set()
        # Deep copy needed
        for edge in connections:
            self.connect(start_state, end_state, edge.copy())

    def _num_inputs(self, state: State,
                    exclude_self: bool = False) -> int:
        """
        Count the number of transitions which are inputs to the state.

        Arguments:
            state -- Which state to count inputs for.
            exclude_self -- Whether self-loops should be ignored

        Returns:
            The amount of inputs to the given state
        """
        return (np.sum(np.vectorize(len)(self.edge_map[:, state]))
                - (len(self.edge_map[state, state])
                    if exclude_self else 0)
                # Phantom input to start state
                + (state == self.start))

    def _num_outputs(self, state: State,
                     exclude_self: bool = False) -> int:
        """
        Count the number of transitions which are outputs to the state.

        Arguments:
            state -- Which state to count outputs for.
            exclude_self -- Whether self-loops should be ignored

        Returns:
            The amount of outputs to the given state
        """
        return (np.sum(np.vectorize(len)(self.edge_map[state, :]))
                - (len(self.edge_map[state, state])
                    if exclude_self else 0)
                # Phantom ouput from end state
                + (state == self.end))

    def _remove_if_unreachable(self, state: State) -> bool:
        """
        Removes the given state in the case that it either has no inputs
        (is unreachable) or has no outputs (dead-end).

        Arguments:
            state -- The state to check for removal.

        Returns:
            Boolean indicating whether or not the state was deleted.
        """
        if (self._num_inputs(state, exclude_self=True) < 1
                or self._num_outputs(state, exclude_self=True) < 1):
            self._remove_state(state)
            return True
        return False

    def _remove_state(self, state: State) -> None:
        """
        Remove a state, deleting it from the edge_map and resizing the
        map to shrink-fit.

        Arguments:
            state -- The state to remove.
        """
        # Adjust indices
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

    def _merge_outputs(self, destination: State, source: State) -> bool:
        """
        Copy the outputs of one state to another.

        Arguments:
            destination -- The state to copy outputs to.
            source -- The state to copy outputs from.

        Returns:
            Whether any actual changes were made during the merge
        """
        assert destination >= 0
        assert destination < self.size
        assert source >= 0
        assert source < self.size
        changed = False
        # Iterate s2 row and make same connections from s1
        for index, src_set in enumerate(self.edge_map[source, :]):
            dst_set = self.edge_map[destination, index]
            if changed:
                pass  # Fast path, don't perform more comparisons
            elif (len(src_set) > len(dst_set)):
                # More elements, definately changed
                changed = True
            elif ParserPredicate.set_mutable_diff(src_set, dst_set):
                changed = True
            else:  # Skip merge if no changes made (small optimisation)
                continue
            self.connect_many(destination, index, src_set)
        return changed

    def _merge_inputs(self, destination: State, source: State) -> bool:
        """
        Copy the inputs of one state to another.

        Arguments:
            destination -- The state to copy inputs to.
            source -- The state to copy inputs from.

        Returns:
            Whether any actual changes were made during the merge
        """
        assert destination >= 0
        assert destination < self.size
        assert source >= 0
        assert source < self.size
        changed = False
        # Iterate s2 column and make same connections to s1
        for index, src_set in enumerate(self.edge_map[:, source]):
            dst_set = self.edge_map[index, destination]
            if changed:
                pass  # Fast path, don't perform more comparisons
            elif (len(src_set) > len(dst_set)):
                # More elements, definately changed
                changed = True
            elif ParserPredicate.set_mutable_symdiff(src_set, dst_set):
                changed = True
            else:  # Skip merge if no changes made (small optimisation)
                continue
            self.connect_many(index, destination, src_set)
        return changed

    def _merge(self, destination: State, source: State) -> None:
        """
        Copy the inputs and outputs of one state to another.

        Arguments:
            destination -- The state to copy to.
            source -- The state to copy from.
        """
        self._merge_inputs(destination, source)
        self._merge_outputs(destination, source)

    def _diagonal_block_with(self, other: np.ndarray) -> None:
        """
        Copy one matrix to be diagonally adjacent with our edge_map, so
        that the states of both are available, without creating any
        conflicts or unexpected transitions.

        Arguments:
            other -- The other matrix to embed in ourselves.
        """
        # constructs block matrix like:
        # [[self,  empty]
        #  [empty, other]]
        self.edge_map = np.block([
            [self.edge_map, Regex._empty_arr((self.size,
                                              other.shape[1]))],
            [Regex._empty_arr((other.shape[0], self.size)), other]])

    def __iadd__(self, other: Any) -> Self:
        """
        Appends another Regex to this one, so that to match, an input
        must match this regular expression, and the remaining chars
        should match with the other regular expression (i.e. should
        match both sequentially).

        Arguments:
            other -- The other Regex (or single transition) to append.

        Returns:
            The current object instance.
        """
        if isinstance(other, Regex):
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
            return NotImplemented
        return self

    def __add__(self, other: Any) -> 'Regex':
        """
        Concatenates two Regex, so that to match, an input must match
        the first regular expression, and the remaining chars should
        match with the second regular expression (i.e. should match both
        sequentially).

        Arguments:
            other -- The other regex to add to the current (first) regex

        Returns:
            A new Regex object, the result of the concatenation.
        """
        result = self.copy()
        result += other
        return result

    def __imul__(self, scalar: int) -> Self:
        """
        Repeats the current regular expression {scalar} times, so that
        an input needs to match that many times in sequence to fully
        match.

        Arguments:
            scalar -- The amount of times to repeat the regular
                expression.

        Raises:
            ValueError: If the given value is negative.

        Returns:
            The current Regex instance.
        """
        if not isinstance(scalar, int):
            return NotImplemented
        if scalar == 0:
            self.edge_map = Regex._empty_arr((1, 1))
            self.start = self.end = 0
        elif scalar > 0:
            monomer = self.copy()
            for _ in range(scalar - 1):
                self += monomer.copy()
        else:
            raise ValueError(f"cannot multiply {self} by {scalar}")
        return self

    def __mul__(self, scalar: int) -> 'Regex':
        """
        Repeats the regular expression {scalar} times, so that an input
        needs to match that many times in sequence to fully match.

        Arguments:
            scalar -- The amount of times to repeat the regular
                expression.

        Raises:
            ValueError: If the given value is negative.

        Returns:
            A new Regex, representing the repeated regular expression.
        """
        result = self.copy()
        result *= scalar
        return result

    def __ior__(self, other: 'Regex') -> Self:
        """
        Unionises the other Regex with the current Regex, so that to
        match, an input string can match either the current or the other
        Regex.

        Arguments:
            other -- The other Regex to unionise with.

        Returns:
            The current Regex instance.
        """
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
        """
        Creates the union of the two Regex, so that to match, an input
        string can match either of the two Regex.

        Arguments:
            other -- The other Regex to unionise with.

        Returns:
            A new Regex, the result of the union.
        """
        result = self.copy()
        result |= other
        return result

    def reverse(self) -> 'Regex':
        """
        Reverses the regex, so that it matches the reversed strings.

        Returns:
            A new Regex which matches the reverse strings.
        """
        if self._reverse is None:
            self._prepare_full_reverse()
        assert self._reverse is not None
        return self._reverse.copy()
    __neg__ = reverse

    def _basic_reverse(self) -> 'Regex':
        """
        Performs the basic operations needed for reversal

        Returns:
            A new, reversed Regex
        """
        result = self.copy()
        result.edge_map = result.edge_map.transpose()
        result.start = self.end
        result.end = self.start
        return result

    def _prepare_full_reverse(self) -> None:
        # Need to access other Regex instances
        # pylint: disable=protected-access
        # Import here to avoid cyclic dependency due to type-annotating
        # in regex_optimiser.py
        # pylint: disable-next=import-outside-toplevel
        from .regex_optimiser import _OptimiseRegex
        assert self._base is not None
        reverse = self._base._basic_reverse()
        _OptimiseRegex(reverse)
        reverse._base = reverse.copy()
        reverse.connect(reverse.start,
                        reverse.start,
                        MatchConditions.consume_any)
        _OptimiseRegex(reverse)
        reverse._reverse = self
        self._reverse = reverse

    def __bool__(self) -> bool:
        """
        Returns:
            Whether the regex has any states or transitions.
        """
        return self.size > 1 or bool(self.edge_map[0, 0])

    def is_in(self, value: str) -> bool:
        """
        Tries to match the current regex to the given string.

        Arguments:
            value -- The string to match against.

        Returns:
            Whether the Regex was found in the string.
        """
        ctx = MatchConditions(value)
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
    test = is_in

    def _match_index(self, value: str, *, start: int = 0) -> int:
        """
        Finds the index where this regex finds a match within the string

        Arguments:
            value -- The string to search

        Keyword Arguments:
            start -- The index of the string to start at (default: {0})

        Returns:
            The index of the last char within the found substring (-1
            for no match)
        """
        # Note implementation is very similar to is_in, but is slightly
        # more featured, because it not only needs to find the match,
        # but also the last index of the match. As such it does not end
        # right away when it hits the end state, but instead searches to
        # see if it could find any more
        ctx = MatchConditions(value)
        # pylint: disable-next=protected-access
        ctx._cursor = start
        state = self.start
        end_idx: int = -1
        while True:
            if state == self.start:
                # "temp" workaround for not missing entire match
                if end_idx > 0:
                    return end_idx
            if state == self.end:
                # pylint: disable-next=protected-access
                end_idx = ctx._cursor
            exit_state: Optional[State] = None
            for i in range(self.size):
                for edge in self.edge_map[state, i]:
                    edge: ParserPredicate
                    if edge == MatchConditions.epsilon_transition:
                        # e-moves are low-priority, save for later
                        exit_state = i
                    elif edge.evaluate(ctx):
                        state = i
                        break  # outer loop
                else: # cursedness to break to outer loop
                    continue
                break
            else:
                if exit_state is not None:
                    state = exit_state
                    continue # outer loop
                # No match, return last found end index
                return end_idx

    def match(self, value: str) -> Iterable[slice[int]]:
        self._prepare_full_reverse()
        starts, ends = [], []
        idx = 0
        while (idx := self._match_index(value, start=idx)) >= 0:
            ends.append(idx)
        self._prepare_full_reverse()
        assert self._reverse is not None
        str_reverse = value[::-1] # cursed
        idx = 0
        # Dont use .reverse() to avoid unnecesary copy
        # pylint: disable-next=protected-access
        while (idx := self._reverse._match_index(str_reverse,
                                                 start=idx)) >= 0:
            starts.append(idx)
        # TODO:

    def optional(self) -> Self:
        """
        Makes the current Regex optional, so that either a matching
        string or an empty string can match.

        Returns:
            The current Regex instance.
        """
        self.connect(self.start, self.end,
                     MatchConditions.epsilon_transition)
        return self

    def repeat(self) -> Self:
        """
        Makes the current Regex repeated, so that it will match any
        number of sequential matching strings.

        Returns:
            The current Regex instance.
        """
        self.connect(self.end, self.start,
                     MatchConditions.epsilon_transition)
        return self

    def copy(self):
        """
        Creates a clone of the current regex.

        Returns:
            A new Regex instance identical to this one.
        """
        # Delegate to copy constructor
        return Regex(self)

    def __str__(self) -> str:
        """
        Creates pretty-printable representation of the current Regex
        transition table, useful for debugging.

        Returns:
            The formatted string.
        """
        inner_arrs = ',\n '.join([
            "[" + ', '.join([
                f"{{{', '.join([str(edge) for edge in edges])}}}"
                if isinstance(edges, set) else "{}"
                for edges in row]) + "]"
            for row in self.edge_map])
        return f"[{inner_arrs}]: {self.start} -> {self.end}"

    def _find_double_refs(self) -> Mapping[int, set[tuple[int, ...]]]:
        """
        Searches the Regex for connections which share a set object,
        i.e. they hold a reference to the same object. Used for
        debugging parts of the builder and optimiser which create, move
        or clone parts of the matrix around

        Returns:
            A map of memory addresses to a set of coordinates which all
            reference to that memory address, excluding the cases where
            there is only one reference to that address
        """
        coord_map: dict[int, set[tuple[int, ...]]] = {}
        it = np.nditer(self.edge_map,
                       flags=['multi_index', 'refs_ok'])
        for el in it:
            # thx numpy [()]
            el_id = id(el[()])  # type: ignore
            if el_id in coord_map:
                coord_map[el_id].add(it.multi_index)
            else:
                coord_map[el_id] = set((it.multi_index,))
        return {k: v for k, v in coord_map.items() if len(v) > 1}
