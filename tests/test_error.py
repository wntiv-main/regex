"""Exceptions for use within tests to report failure"""

__author__ = "Callum Hynes"
__all__ = ["TestError", "TestErrorImpl", "ExceptionAsTestError",
           "RegexBuildError", "StateIdentityError", "EdgeNotFoundError",
           "ExtraEdgesError", "RegexMatchError"]

from abc import ABC, abstractmethod
from enum import IntEnum, auto
from typing import Any, assert_never, override

from regex import Regex
from regex.regexutil import ParserPredicate, State

from . import regex_tests


class TestError(Exception, ABC):
    """An exception to be raised if a test case fails"""

    @abstractmethod
    def outcome_message(self) -> str:
        """
        Descriptive details about the error to be included in the final
        test results table

        Returns:
            String description of the error
        """
        raise NotImplementedError()

    def __str__(self) -> str:
        """
        String representation of the exception for debuggers, and in
        case the exception should escape the test package

        Returns:
            Detailed string representation of the exception
        """
        result = (f"{self.__class__.__name__} during testing: "
                  f"{self.outcome_message()}")
        if self.__cause__ is not None:
            result += (
                f"\nCaused by {self.__cause__.__class__.__name__}: "
                f"{self.__cause__}")
        return result


class TestErrorImpl(TestError):
    """Implementation of TestError with pre-determined error message"""

    _msg: str
    """The error message to output"""

    def __init__(self, msg: str):
        super().__init__(msg)
        self._msg = msg

    @override
    def outcome_message(self) -> str:
        return f"Test failed: {self._msg}"


class ExceptionAsTestError(TestError):
    """
    Wrapper for another exception which should be treated like a
    TestError
    """

    @override
    def outcome_message(self) -> str:
        return (f"{self.__cause__.__class__.__name__} was raised: "
                f"{self.__cause__}")


class RegexBuildError(TestError):
    """A test error related to the way a Regex was built"""


class StateIdentityError(RegexBuildError):
    """Error indicating a state was not of the intended type"""

    _state: 'regex_tests.NodeMatcher'
    """The state which caused the error"""

    _attempted_state: 'regex_tests.RegexState'
    """The type that the state was expected to be"""

    def __init__(self,
                 state: 'regex_tests.NodeMatcher',
                 tried_to_be: 'regex_tests.RegexState'):
        super().__init__(state, tried_to_be)
        self._state = state
        self._attempted_state = tried_to_be

    @override
    def outcome_message(self) -> str:
        state_name: str
        match self._attempted_state:
            case regex_tests.RegexState.END:
                state_name = "the end state"
            case regex_tests.RegexState.START:
                state_name = "the start state"
            case (regex_tests.RegexState.ANY
                  | regex_tests.RegexState.SELF):
                # Unreachable
                assert False, "States should always be ANY and SELF"
            case _:
                assert_never(self._attempted_state)
        return f"State {self._state.state_id} was not {state_name}."


class EdgeNotFoundError(RegexBuildError):
    """
    Error indicating that an edge was expected to be on a state, but was
    not present upon inspection
    """

    _start_state: 'regex_tests.NodeMatcher'
    """The state where the edge should have started"""

    _end_state: 'regex_tests.NodeMatcher'
    """The state where the edge should have ended"""

    _expected_edge: ParserPredicate
    """The edge which should of connected the two states"""

    def __init__(self,
                 state: 'regex_tests.NodeMatcher',
                 edge: ParserPredicate,
                 to: 'regex_tests.NodeMatcher'):
        super().__init__(state, edge, to)
        self._start_state = state
        self._end_state = to
        self._expected_edge = edge

    @override
    def outcome_message(self) -> str:
        result = (f"Did not find {self._expected_edge}-move from "
                  f"{self._start_state.state_id}")
        if self._end_state.state_id is not None:
            result += f" to {self._end_state.state_id}"
        result += '.'
        return result


class ExtraEdgesError(RegexBuildError):
    """Indicates that unexpected extra edges were present on a state"""

    _start_state: 'regex_tests.NodeMatcher'
    """The state which the extra edges were found on"""

    _extras: dict[State, list[ParserPredicate]]
    """The extra edges that were found"""

    def __init__(self,
                 state: 'regex_tests.NodeMatcher',
                 extras: dict[State, list[ParserPredicate]]):
        super().__init__(state, extras)
        self._start_state = state
        self._extras = extras

    @override
    def outcome_message(self) -> str:
        result = (f"Found unexpected edges from state "
                  f"{self._start_state.state_id}:")
        for state, edges in self._extras.items():
            result += (
                f"\n- {', '.join(map(lambda x: f'{x}-move', edges))} "
                f"to state {state}")
        return result


class RegexMatchError(TestError):
    """
    Error indicating that a regex that was expected to (not) match
    did(n't) match
    """

    _regex: Regex
    """The regex which was being matched against"""

    _match: str
    """The string that was being tested"""

    _should_match: bool
    """Whether the string was expected to match or not"""

    def __init__(self,
                 regex: Regex,
                 match: str,
                 should_match: bool):
        super().__init__(regex, match, should_match)
        self._regex = regex
        self._match = match
        self._should_match = should_match

    @override
    def outcome_message(self, *, _nt="n't") -> str:
        return (f"'{self._match}' "
                f"did{_nt if self._should_match else ''} match the "
                f"regular expression.")


class RegexPositionalMatchError(TestError):
    """
    Error indicating that a regex that was expected to match a specific
    substring at the specified locaation did not match
    """

    class Type(IntEnum):
        """The specific type of the error"""
        NO_MATCH = auto()
        MATCH_MISSED = auto()
        INCORRECT_SUBSTRING = auto()

    _regex: Regex
    """The regex which was being matched against"""

    _source: str
    """The string that was being tested"""

    _match_position: slice
    """The expected position of the match, and expected substring"""
    # Abusing `step` member to store expected substring

    _extra: Any | None

    _error_type: Type
    """The specific type of the error"""

    def __init__(self, # pylint: disable=too-many-arguments
                 regex: Regex,
                 source: str,
                 test: slice,
                 error_type: Type,
                 got: Any | None = None):
        super().__init__(regex, source, test, error_type)
        self._regex = regex
        self._source = source
        self._match_position = test
        self._error_type = error_type
        self._extra = got

    @override
    def outcome_message(self) -> str:
        Type = RegexPositionalMatchError.Type
        match self._error_type:
            case Type.NO_MATCH:
                return (f"No match was found at "
                        f"{self._match_position.start}:"
                        f"{self._match_position.stop}"
                        + f" (expecting: '{self._match_position.step}')"
                            if self._match_position.step is not None
                            else '')
            case Type.MATCH_MISSED:
                return (f"Match was found at "
                        f"{self._match_position.start}, but did not end"
                        f" at {self._match_position.stop} as expected")
            case Type.INCORRECT_SUBSTRING:
                return (f"Match was found at "
                        f"{self._match_position.start}:"
                        f"{self._match_position.stop}, but did not "
                        f"match the expected substring: "
                        f"'{self._match_position.step}' (got: "
                        f"'{self._extra}')")
            case _:
                assert_never(self._error_type)
