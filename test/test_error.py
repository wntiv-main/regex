from abc import ABC, abstractmethod

from src.regexutil import ParserPredicate, State
from . import regex_matcher


class TestError(Exception, ABC):
    @abstractmethod
    def outcome_message(self) -> str:
        raise NotImplementedError()

    def __str__(self) -> str:
        result = (f"{self.__class__.__name__} during testing: "
                  f"{self.outcome_message()}")
        if self.__cause__:
            result += (
                f"\nCaused by {self.__cause__.__class__.__name__}: "
                f"{self.__cause__}")


class RegexTestError(TestError):
    pass


class StateIdentityError(RegexTestError):
    _state: 'regex_matcher.NodeMatcher'
    _attempted_state: 'regex_matcher.RegexState'

    def __init__(self,
                 state: 'regex_matcher.NodeMatcher',
                 tried_to_be: 'regex_matcher.RegexState'):
        super().__init__(state, tried_to_be)
        self._state = state
        self._attempted_state = tried_to_be

    def outcome_message(self) -> str:
        state_name: str
        match self._attempted_state:
            case regex_matcher.RegexState.END:
                state_name = "the end state"
            case regex_matcher.RegexState.START:
                state_name = "the start state"
            case _:
                raise ValueError(
                    f"Unexpected Enum value: {self._attempted_state}")
        return f"State {self._state._for} was not {state_name}."


class EdgeNotFoundError(RegexTestError):
    _start_state: 'regex_matcher.NodeMatcher'
    _end_state: State | None
    _expected_edge: ParserPredicate

    def __init__(self,
                 state: 'regex_matcher.NodeMatcher',
                 edge: ParserPredicate,
                 to: State | None = None):
        super().__init__(state, edge, to)
        self._start_state = state
        self._end_state = to
        self._expected_edge = edge

    def outcome_message(self) -> str:
        result = (f"Did not find {self._expected_edge}-move from "
                  f"{self._start_state._for}")
        if self._end_state is not None:
            result += f" to {self._end_state}"
        result += '.'
        return result


class ExtraEdgesError(RegexTestError):
    _start_state: 'regex_matcher.NodeMatcher'
    _extras: dict[State, list[ParserPredicate]]

    def __init__(self,
                 state: 'regex_matcher.NodeMatcher',
                 extras: dict[State, list[ParserPredicate]]):
        super().__init__(state, extras)
        self._start_state = state
        self._extras = extras

    def outcome_message(self) -> str:
        result = (f"Found unexpected edges from state "
                  f"{self._start_state._for}:")
        for state, edges in self._extras.items():
            result += (
                f"\n- {', '.join(map(lambda x: f'{x}-move', edges))} "
                f"to state {state}")
        return result
