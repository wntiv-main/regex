"""Utilities for performing tests on Regex"""

__author__ = "Callum Hynes"
__all__ = ["RegexState", "NodeMatcher", "TestRegexShape",
           "TestRegexMatches", "TestNoParseError", "TestParseError"]

import sys
from enum import IntEnum, auto
from typing import Callable, Literal, Self, assert_never, override

from regex import Regex
from regex.regex_factory import PatternParseError
from regex.regexutil import (ConsumeAny, ConsumeString, ParserPredicate,
                             SignedSet, State)

from .test import (AssertNoRaises, AssertRaises, ResultType, TestCase,
                   TestType)
from .test_error import (EdgeNotFoundError, ExtraEdgesError,
                         RegexMatchError, RegexPositionalMatchError,
                         StateIdentityError)

if __debug__ and not '--headless' in sys.argv:
    from regex.debug_graph_viewer import (DebugGraphViewer,
                                          MultiFigureViewer)



class RegexState(IntEnum):
    """Discriminates some unique states from the DFA"""
    START = auto()
    SELF = auto()
    END = auto()
    ANY = auto()


class NodeMatcher:
    """
    Represents the decription of a state (node) of the DFA, which is
    later fitted against an actual DFA to check if the entire graph
    matches the description described by the NodeMatchers
    """

    _handler: 'TestRegexShape'
    """
    The parent TestCase, which holds a reference to the Regex to match
    """

    _children: list[tuple[ParserPredicate, 'NodeMatcher']]
    """Description of the expected edges leaving this state"""

    state_id: State | None
    """The state ID that this node has been matched with"""

    _evaluated: bool
    """Whether this node has been matched up to a state yet"""

    _type: Literal[RegexState.ANY, RegexState.START, RegexState.END]
    """The unique type of the state, ANY if not a special state"""

    _expected_type: Literal[RegexState.ANY,
                            RegexState.START, RegexState.END]
    """The state type that this state is also expected to be"""

    _replaced_by: 'NodeMatcher | None'
    """The actual NodeMatcher, in case this must act as a proxy"""

    def __init__(self,
                 handler: 'TestRegexShape',
                 state_type: RegexState = RegexState.ANY):
        self._handler = handler
        self._children = []
        self.state_id = None
        self._evaluated = False
        match state_type:
            case RegexState.ANY | RegexState.START | RegexState.END:
                # UPDATE _type TYPE ANNOTATION WHEN MODIFIED
                self._type = state_type
            case RegexState.SELF:
                raise ValueError("state_type cannot be SELF")
            case _:
                assert_never(state_type)
        self._expected_type = RegexState.ANY
        self._replaced_by = None

    def _state_name(self) -> str:
        """
        Context-dependent human-readable description of this state

        Returns:
            The description of this state
        """
        result: str
        match self._type:
            case RegexState.START:
                result = "the start state"
            case RegexState.ANY:
                result = "state"
            case RegexState.END:
                result = "the end state"
            case _:
                assert_never(self._type)
        match self._expected_type:
            case RegexState.START:
                result += "(which is also the start state)"
            case RegexState.END:
                result = "(which is also the end state)"
            case RegexState.ANY:
                pass
            case _:
                assert_never(self._expected_type)
        if self.state_id is not None:
            result += f" ({self.state_id})"
        elif result == "state":
            result = "a state"
        return result

    def _replace(self, other: 'NodeMatcher') -> None:
        """
        Take the place of the other node, replacing it with a proxy to
        ourself. Also copy its children, if any

        Arguments:
            other -- The state to impersonate
        """
        # pylint: disable=protected-access
        for edge, state in other._children:
            self._children.append((edge, state))
        other._replaced_by = self

    def is_also(self,
                state: Literal[RegexState.START,
                               RegexState.END]) -> Self:
        """
        Asserts that this state is also an instance of the given state
        type. If not, the test will fail when the NodeMatcher tries to
        match onto the DFA

        Arguments:
            state -- The state type that this state is expected to be

        Returns:
            The current instance
        """
        # pylint: disable=protected-access
        if state == RegexState.END and self._handler._end is None:
            self._handler._end = self
            self._type = RegexState.END
        elif state == RegexState.END and self is not self._handler._end:
            # see above, typechecker :|
            assert self._handler._end is not None
            self._replace(self._handler._end)
            self._handler._end = self
            self._type = RegexState.END
        elif (state == RegexState.START
                and self is not self._handler._start):
            self._replace(self._handler._start)
            self._handler._start = self
            self._type = RegexState.START
        else:
            self._expected_type = state
        return self

    def has_literal(self, literal_match: str,
                    to: 'NodeMatcher | RegexState' = RegexState.ANY)\
            -> 'NodeMatcher':
        """
        Asserts that this node has a ConsumeString move to another state
        of the given type. An error will be raised during evaluation if
        this is not the case

        Arguments:
            literal_match -- The string to match
            to -- The type of the next state (default: {RegexState.ANY})

        Returns:
            The next state
        """
        return self.has(ConsumeString(literal_match), to)

    def has_literal_chain(self, literal_chain: str,
                          to: 'NodeMatcher | RegexState'
                          = RegexState.ANY) -> 'NodeMatcher':
        """
        Utility function for asserting an entire chain of single-char
        ConsumeString edges between states, in sequence

        Arguments:
            literal_chain -- The string to match
            to -- The type of the last state (default: {RegexState.ANY})

        Returns:
            The last state in the chain
        """
        last = self
        # Skip last in loop to pass `to` arg
        for char in literal_chain[:-1]:
            last = last.has_literal(char)
        return last.has_literal(literal_chain[-1], to)

    def has_any(self, of: str,
                to: 'NodeMatcher | RegexState' = RegexState.ANY)\
            -> 'NodeMatcher':
        """
        Asserts that the current state has a ConsumeAny move to the next
        state, which accepts any char in the given string

        Arguments:
            of -- The chars which will be accepted
            to -- The type of the next state (default: {RegexState.ANY})

        Returns:
            The next state
        """
        return self.has(ConsumeAny(SignedSet(of)), to)

    def has_any_except(self, of: str,
                       to: 'NodeMatcher | RegexState' = RegexState.ANY)\
            -> 'NodeMatcher':
        """
        Asserts that the current state has a ConsumeAny move to the next
        state, which accepts any char NOT in the given string

        Arguments:
            of -- The chars which will NOT be accepted
            to -- The type of the next state (default: {RegexState.ANY})

        Returns:
            The next state
        """
        return self.has(ConsumeAny(SignedSet(of, True)), to)

    def has(self, edge: ParserPredicate,
            to: 'NodeMatcher | RegexState' = RegexState.ANY)\
            -> 'NodeMatcher':
        """
        Asserts that the current state has the given edge to a state of
        the type specified

        Arguments:
            edge -- The predicate to expect
            to -- The type of the next state (default: {RegexState.ANY})

        Returns:
            The next state
        """
        # Proxy
        if self._replaced_by is not None:
            return self._replaced_by.has(edge, to)
        next_state: NodeMatcher
        # pylint: disable=protected-access
        match to:
            case RegexState.ANY:
                next_state = NodeMatcher(self._handler)
            case RegexState.START:
                next_state = self._handler._start
            case RegexState.END:
                if self._handler._end is None:
                    self._handler._end = NodeMatcher(self._handler,
                                                     RegexState.END)
                next_state = self._handler._end
            case RegexState.SELF:
                next_state = self
            case NodeMatcher() as x:
                next_state = x
            case _:
                assert_never(to)
        self._children.append((edge, next_state))
        return next_state

    @staticmethod
    # pylint: disable-next=dangerous-default-value
    def _num_ending(num: int, *,
                    _ENDINGS=['th', 'st', 'nd', 'rd']) -> str:
        """
        Applies the correct suffix to the given number

        Arguments:
            num -- The number to decorate

        Returns:
            The string decoration to be appended to the number
        """
        if 10 <= num < 20 or num % 10 > 3:
            return 'th'
        return _ENDINGS[num % 10]

    @staticmethod
    def _num_w_ending(num: int) -> str:
        """
        Decorates the given number with the correct suffix

        Arguments:
            num -- The number to decorate

        Returns:
            The string decoration, appended to the number
        """
        return f"{num}{NodeMatcher._num_ending(num)}"

    def _msg(self,  # pylint: disable=design
             _visited: list['NodeMatcher'],
             _indent: int = 0,
             _top: bool = False,
             *, _TAB='\t') -> str:
        """
        Cursed recursive logic to produce a human-readable description
        of the "multi-digraph" structure.

        Arguments:
            _visited -- list of previously visited nodes
            _top -- special handling for intermediate, joining words
                (default: {False})

        Returns:
            A string description of this node in the graph.
        """
        # pylint: disable=protected-access
        if self._replaced_by is not None:
            # Proxy other
            return self._replaced_by._msg(_visited, _indent, _top)
        joiner = ""
        if not _top:
            joiner = " to "
        if self in _visited:
            if self._type == RegexState.END:
                return f"{joiner}the end state."
            if self._type == RegexState.START:
                return f"{joiner}the start state."
            left_idx = _visited.index(self)
            right_idx = len(_visited) - left_idx
            relative_pos: str
            if right_idx == 1:
                relative_pos = "current"
            elif right_idx == 2:
                relative_pos = "previous"
            elif left_idx == 0:
                # If we get here *something* is wrong, but ill put this
                # here anyway ig...
                relative_pos = "start"
            elif left_idx > right_idx:
                relative_pos = (NodeMatcher._num_w_ending(right_idx - 1)
                                + " previous")
            else:
                relative_pos = NodeMatcher._num_w_ending(left_idx + 1)
            return (f"{joiner.replace('to', 'back to')}the "
                    f"{relative_pos} state.")
        _visited.append(self)
        if not self._children:
            return f"{joiner}{self._state_name()}."
        if len(self._children) == 1:
            result = ""
            if _top:
                result = self._state_name()
            if self._type != RegexState.ANY:
                result = f"{joiner}{self._state_name()}"
            result += f", followed by an {self._children[0][0]}-move"
            result += self._children[0][1]._msg(_visited, _indent)
            return result
        result = joiner
        result += f"{self._state_name()}, followed by:"
        for move, child in self._children:
            result += (f"\n{_TAB * _indent}- an {move}-move to "
                       f"{child._msg(_visited, _indent + 1, True)}")
        return result

    def _evaluate(self, _taken: set[State]) -> None:
        """
        Attempts to match up the current node description to an actual
        state in the Regex

        Arguments:
            _taken -- Set of state IDs which have already been matched

        Raises:
            StateIdentityError: If this state cannot be of the expected
                type
            EdgeNotFoundError: If any of the edges expected to leave
                this state are not present
            ExtraEdgesError: If there are any extra edges leaving this
                state which are not expected to be present
        """
        # pylint: disable=protected-access
        if self._replaced_by is not None:
            # Proxy other
            self._replaced_by._evaluate(_taken)
            return
        assert self._handler._regex is not None
        if self._evaluated:
            return  # Already been visited
        self._evaluated = True
        match self._expected_type:
            case RegexState.END:
                if self.state_id != self._handler._regex.end:
                    raise StateIdentityError(self, self._expected_type)
            case RegexState.START:
                if self.state_id != self._handler._regex.start:
                    raise StateIdentityError(self, self._expected_type)
            case RegexState.ANY | RegexState.SELF:
                pass
            case _:
                assert_never(self._type)
        for edge, child in self._children:
            if child.state_id is None:
                match child._type:
                    case RegexState.START:
                        child.state_id = self._handler._regex.start
                    case RegexState.END:
                        child.state_id = self._handler._regex.end
                    case RegexState.ANY:
                        for state in range(self._handler._regex.size):
                            if (state in _taken
                                or state == self._handler._regex.start
                                    or state == self._handler._regex.end):
                                # State already has NodeMatcher
                                continue
                            if edge.kind_of_in(self._handler._regex
                                               .edge_map[self.state_id,
                                                         state]):
                                child.state_id = state
                                _taken.add(state)
                                break
                        else:
                            raise EdgeNotFoundError(self, edge, child)
                    case _:
                        assert_never(child._type)
            # cursedness since ParserPredicates are mutable
            found_edge = edge.kind_of_in(self._handler._regex.edge_map[
                self.state_id, child.state_id])
            if found_edge is None:
                # Connection not found
                raise EdgeNotFoundError(self, edge, child)
            self._handler._regex.edge_map[
                self.state_id,
                child.state_id].remove(found_edge)  # type: ignore
            child._evaluate(_taken)
        if self._handler._regex.edge_map[self.state_id, :].any():
            extras = {
                state: list(self._handler._regex
                            .edge_map[self.state_id, state])
                for state in range(self._handler._regex.size)}
            raise ExtraEdgesError(self, extras)


# pylint: disable-next=too-many-instance-attributes
class TestRegexShape(TestCase):
    """
    A test case requiring the regex generated from a given pattern to
    match the "shape" described. This class is used as a decorator, so
    that the decorated function is able to describe the expected "shape"
    of the Regex
    """
    if __debug__ and not '--headless' in sys.argv:
        _failed_regex: MultiFigureViewer = MultiFigureViewer()
        """A viewer containing diagrams of all the failed test cases"""

    _pattern: str
    """The Regex pattern to use"""

    _reverse: bool
    """Whether to test the reversed regex instead"""

    _regex: Regex | None
    """The Regex object to be matched against"""

    _debug_regex: Regex | None
    """A copy of the Regex for the debug viewer"""

    _start: NodeMatcher
    """The matcher for the starting node in the Regex"""

    _end: NodeMatcher | None
    """The matcher for the ending node in the Regex"""

    _expecting: str | None
    """An optional, more detailed description of the expected outcome"""

    def __init__(self, pattern: str,
                 test_type: TestType = TestType.EXPECTED,
                 expecting: str | None = None,
                 reverse: bool = False):
        description = (f"Trying to construct regular expression from "
                       f"`{pattern}`")
        if reverse:
            description += ", and reverse it"
        super().__init__(test_type, description)
        self._start = NodeMatcher(self, RegexState.START)
        # Not initialized yet as maybe == _start?
        self._end = None
        self._pattern = pattern
        self._reverse = reverse
        self._expecting = expecting

    # For use as decorator
    def __call__(self, func: Callable[[NodeMatcher], None]) -> None:
        """
        Initilize the test with the Regex "shape" described

        Arguments:
            func -- A callable that should accept the starting node
                matcher and describe how further nodes and edges are
                related, describing the "shape" of the expected Regex
        """
        func(self._start)
        self._description = (
            f"{func.__name__.replace('_', ' ').capitalize()}: "
            f"{self._description}")

    @override
    def _inner_test(self) -> None:
        """
        Tries to match the shape described to the actual Regex produced,
        and updates the test case fields to describe the results
        """
        # pylint: disable=protected-access
        # Initialize _regex here so errors are catched by test
        self._regex = Regex(self._pattern)
        if self._reverse:
            self._regex = self._regex.reverse()
        self._debug_regex = self._regex.copy()
        if self._expecting is not None:
            self.set_expected(
                f"Expected {self._expecting}. As such the DFA should "
                f"start with {self._start._msg([], _top=True)}")
        else:
            self.set_expected(
                f"Expected a DFA to be produced with "
                f"{self._start._msg([], _top=True)}")
        self._evaluate()
        self.set_outcome(
            f"Produced a DFA with "
            f"{self._start._msg([], _top=True)}")

    @override
    def run(self):
        super().run()
        # On fail, create debug views for debugging
        if (self._result == ResultType.FAIL
                and __debug__ and not '--headless' in sys.argv):
            assert self._debug_regex is not None
            fig = DebugGraphViewer(self._debug_regex.edge_map,
                                   self._debug_regex.start,
                                   self._debug_regex.end).render()
            fig.suptitle(str(self._outcome), fontsize=8)
            fig.canvas.manager.set_window_title(  # type: ignore
                self._description)
            TestRegexShape._failed_regex.add(fig)

    def _evaluate(self):
        """Evaluate the node matchers on the Regex"""
        assert self._regex is not None
        # pylint: disable=protected-access
        self._start.state_id = self._regex.start
        self._start._evaluate(set((self._regex.end,)))


class TestRegexMatches(TestCase):
    """
    Test case to assert that a given regular expression is successfully
    matched in some expected strings, and not matched in given
    unexpected strings
    """

    _pattern: str
    """The regular expression to use for matching"""

    _reverse: bool
    """Whether the reverse regex should be tested instead"""

    _regex: Regex | None
    """The Regex compiled from the regular expression pattern"""

    _expected_matches: set[str]
    """The set of strings expected to match"""

    _unexpected_matches: set[str]
    """The set of strings expected to NOT match"""

    def __init__(self, pattern: str,
                 test_type: TestType = TestType.EXPECTED,
                 reverse: bool = False):
        super().__init__(test_type, f"Testing matches for `{pattern}`")
        self._pattern = pattern
        self._reverse = reverse
        # Defer initialization for error capture
        self._regex = None
        self._expected_matches = set()
        self._unexpected_matches = set()

    @override
    def _inner_test(self) -> None:
        """Test regex for strings, update test case outcome fields"""
        expected = "Expected "
        if self._expected_matches:
            if len(self._expected_matches) == 1:
                expected += (f"{self._expected_matches.copy().pop()!r} "
                             f"to match")
            else:
                expected += (f"{self._expected_matches} "
                             f"all to match")
            if self._unexpected_matches:
                expected += ", and "
        if self._unexpected_matches:
            if len(self._unexpected_matches) == 1:
                expected += (f"{self._unexpected_matches.copy().pop()!r} "
                             f"to not match")
            else:
                expected += (f"{self._unexpected_matches} "
                             f"all to not match")
        self.set_expected(expected)
        self._regex = Regex(self._pattern)
        if self._reverse:
            self._regex = self._regex.reverse()
        for test in self._expected_matches:
            if not self._regex.is_in(test):
                raise RegexMatchError(self._regex, test, True)
        for test in self._unexpected_matches:
            if self._regex.is_in(test):
                raise RegexMatchError(self._regex, test, False)

    def assert_matches(self, *tests: str) -> Self:
        """
        Asserts that the Regex matches the given strings

        Returns:
            The current instance
        """
        for test in tests:
            self._expected_matches.add(test)
        return self

    def assert_doesnt_match(self, *tests: str) -> Self:
        """
        Asserts that the Regex does NOT match the given strings

        Returns:
            The current instance
        """
        for test in tests:
            self._unexpected_matches.add(test)
        return self


class TestRegexMatchesAt(TestCase):
    """
    Test case to assert that a given regular expression is matched at
    the correct locations within the given source strings
    """

    _pattern: str
    """The regular expression to use for matching"""

    _reverse: bool
    """Whether the reverse regex should be tested instead"""

    _regex: Regex | None
    """The Regex compiled from the regular expression pattern"""

    _expected_matches: dict[str, tuple[slice, ...]]
    """The set of strings expected to match"""

    class _Helper: # pylint: disable=too-few-public-methods
        """
        Handles asserting expected matches, using a clean(ish) syntax by
        allowing slice notation (in a slightly cursed way)
        """

        _parent: 'TestRegexMatchesAt'
        """The TestCase to append this test to"""

        _source_str: str
        """The string to search for the given matches"""

        def __init__(self, parent: 'TestRegexMatchesAt',
                     source: str):
            self._parent = parent
            self._source_str = source

        def __getitem__(
            self,
            match_positions: slice
                           | tuple[slice, ...]) -> 'TestRegexMatchesAt':
            """
            Asserts that the Regex matches at this location in the
            source string, and finds the right substring

            Arguments:
                match_positions -- The positions of the expected matches
                    as slices, optionally with the expected substring to
                    match as the `step` parameter: i.e., one might write
                    test.assert_matches_at('...')[3:8:"hello", ...etc]

            Returns:
                The parent test case, for chaining
            """
            if isinstance(match_positions, slice):
                match_positions = (match_positions,)
            self._parent._expected_matches[
                self._source_str] = match_positions
            return self._parent

    def __init__(self, pattern: str,
                 test_type: TestType = TestType.EXPECTED,
                 reverse: bool = False):
        super().__init__(test_type, f"Testing matches for `{pattern}`")
        self._pattern = pattern
        self._reverse = reverse
        # Defer initialization for error capture
        self._regex = None
        self._expected_matches = {}

    @override
    def _inner_test(self) -> None:
        """Test strings for matches, update test case outcome fields"""
        first = True
        expected = "Expected to find matches"
        for value, matches in self._expected_matches.items():
            if first:
                first = False
            else:
                expected += "; and"
            expected += f" in ```{value}``` at "
            expected += ', '.join(
                f"{idx.start}:{idx.stop}" + (
                    f" (should match `{idx.step}`)"
                    if idx.step is not None else '')
                for idx in matches)
        self.set_expected(expected)
        self._regex = Regex(self._pattern)
        if self._reverse:
            self._regex = self._regex.reverse()
        for source, matches in self._expected_matches.items():
            results = list(self._regex.match(source))
            for idx in matches:
                for i, pos in enumerate(results):
                    if (idx.start is not None
                            and idx.start != pos[0].start):
                        continue
                    if (idx.stop is not None
                            and idx.stop != pos[0].stop):
                        raise RegexPositionalMatchError(
                            self._regex, source, idx,
                            RegexPositionalMatchError.Type.MATCH_MISSED)
                    if idx.step is not None and idx.step != pos[1]:
                        raise RegexPositionalMatchError(
                            self._regex, source, idx,
                            RegexPositionalMatchError
                                .Type.INCORRECT_SUBSTRING,
                            got=pos[1])
                    results.pop(i)
                    break
                else:
                    raise RegexPositionalMatchError(
                        self._regex, source, idx,
                        RegexPositionalMatchError.Type.NO_MATCH)

    def assert_matches_at(self, source: str) -> _Helper:
        """
        Asserts that the Regex matches the given string at the given
        positions

        Arguments:
            source -- The string to search for matches within

        Returns:
            A helper object allowing you to specify the position of
            expected matches, and optionally the expected substring
        """
        return TestRegexMatchesAt._Helper(self, source)


class TestNoParseError(AssertNoRaises):
    """
    Test case that the regular expression builder can successfully
    compile the given regular expression
    """

    _pattern: str
    """The regular expression pattern to try parse/build"""

    def __init__(self, pattern: str,
                 test_type: TestType = TestType.INVALID):
        super().__init__(
            PatternParseError,
            f"Building regular expression from valid pattern "
            f"`{pattern}`",
            test_type)
        self._pattern = pattern
        # Try call constructor
        self._callable = Regex  # type: ignore

    @override
    def _inner_test(self, *args, **kwargs) -> None:
        # Pass pattern, will be passed to Regex constructor
        return super()._inner_test(self._pattern, *args, **kwargs)


class TestParseError(AssertRaises):
    """
    Test case to assert that an appropriate error is raised when trying
    to parse an invalid regular expression pattern
    """

    _pattern: str
    """The pattern to try compile"""

    def __init__(self, pattern: str,
                 test_type: TestType = TestType.INVALID):
        super().__init__(
            PatternParseError,
            f"Building regular expression from invalid pattern "
            f"`{pattern}`",
            test_type)
        self._pattern = pattern
        # Try call constructor
        self._callable = Regex  # type: ignore

    @override
    def _inner_test(self, *args, **kwargs) -> None:
        # Pass pattern, will be passed to Regex constructor
        return super()._inner_test(self._pattern, *args, **kwargs)
