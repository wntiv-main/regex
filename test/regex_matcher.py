from enum import IntEnum, auto
from typing import Self

from src.debug_graph_viewer import DebugGraphViewer, MultiFigureViewer
from src import Regex
from src.regexutil import ConsumeAny, ConsumeString, ParserPredicate, SignedSet, State
from .test import TestCase
from .test_error import EdgeNotFoundError, ExtraEdgesError, StateIdentityError


class RegexState(IntEnum):
    START = auto()
    SELF = auto()
    END = auto()
    ANY = auto()


class NodeMatcher:
    _handler: 'assert_regex'
    _children: list[tuple[ParserPredicate, 'NodeMatcher']]
    _for: State | None
    _type: RegexState
    _expected_type: RegexState

    def __init__(self,
                 handler: 'assert_regex',
                 state_type: RegexState = RegexState.ANY):
        self._handler = handler
        self._children = []
        self._for = None
        self._type = state_type
        self._expected_type = RegexState.ANY

    def _state_name(self) -> str:
        if self._for is not None:
            match self._type:
                case RegexState.START:
                    return f"the start state ({self._for})"
                case RegexState.ANY:
                    return f"state {self._for}"
                case RegexState.END:
                    return f"the end state ({self._for})"
        else:
            match self._type:
                case RegexState.START:
                    return "the start state"
                case RegexState.ANY:
                    return "a state"
                case RegexState.END:
                    return "the end state"
        raise ValueError(f"Unexpected Enum value: {self._type}")

    def is_also(self, state: RegexState) -> Self:
        self._expected_type = state
        return self

    def has_literal(self, literal_match: str,
                    to: 'NodeMatcher | RegexState' = RegexState.ANY)\
            -> 'NodeMatcher':
        return self.has(ConsumeString(literal_match), to)

    def has_any(self, of: str,
                to: 'NodeMatcher | RegexState' = RegexState.ANY)\
            -> 'NodeMatcher':
        return self.has(ConsumeAny(SignedSet(of)), to)

    def has_any_except(self, of: str,
                       to: 'NodeMatcher | RegexState' = RegexState.ANY)\
            -> 'NodeMatcher':
        return self.has(ConsumeAny(SignedSet(of, True)), to)

    def has(self, edge: ParserPredicate,
            to: 'NodeMatcher | RegexState' = RegexState.ANY)\
            -> 'NodeMatcher':
        next: NodeMatcher
        match to:
            case RegexState.ANY:
                next = NodeMatcher(self._handler)
            case RegexState.START:
                next = self._handler._start
            case RegexState.END:
                next = self._handler._end
            case RegexState.SELF:
                next = self
            case NodeMatcher() as x:
                next = x
        self._children.append((edge, next))

    @staticmethod
    def _num_ending(num: int, *,
                    _endings=['th', 'st', 'nd', 'rd']) -> str:
        if (num >= 10 and num < 20) or num % 10 > 3:
            return 'th'
        else:
            return _endings[num % 10]

    @staticmethod
    def _num_w_ending(num: int) -> str:
        return f"{num}{NodeMatcher._num_ending}"

    def _msg(self,
             _visited: list['NodeMatcher'],
             _indent: int = 0,
             _top: bool = False,
             _first: bool = False) -> str:
        if self in _visited:
            left_idx = _visited.index(self)
            right_idx = len(_visited) - left_idx
            relative_pos: str
            if right_idx == 0:
                relative_pos = "previous"
            elif left_idx == 0:
                relative_pos = "start"
            elif left_idx > right_idx:
                relative_pos = (NodeMatcher._num_w_ending(right_idx + 1)
                                + " previous")
            else:
                relative_pos = NodeMatcher._num_w_ending(left_idx + 1)
            return f"back to the {relative_pos} state."
        _visited.append(self)
        if len(self._children) == 0:
            joiner = ""
            if not _top:
                joiner = "to "
            return f"{joiner}{self._state_name()}."
        elif len(self._children) == 1:
            result = ""
            if _top or _first:
                result = self._state_name()
            result += f", followed by an {self._children[0][0]}-move"
            result += self._children[0][1]._msg(_visited, _indent)
            return result
        else:
            result = ""
            if not _top:
                result = "to "
            result += f"{self._state_name()}, followed by any of:"
            for move, child in self._children:
                result += (f"\n{'    ' * _indent}- an {move}-move to "
                           f"{child._msg(_visited, _indent + 1, True)}")
            return result

    def _evaluate(self, _taken: set[State]):
        match self._expected_type:
            case RegexState.END:
                if self._for != self._handler._regex.end:
                    raise StateIdentityError(self, state)
            case RegexState.ANY:
                pass
            case _:
                raise NotImplementedError()
        for edge, child in self._children:
            if child._for is None:
                match child._type:
                    case RegexState.ANY:
                        for state in range(self._handler._regex.size):
                            if state in _taken:
                                continue
                            if edge in (self._handler._regex
                                        .edge_map[self._for, state]):
                                child._for = state
                                _taken.add(state)
                                break
                        else:
                            raise EdgeNotFoundError(self, edge)
                    case RegexState.START:
                        child._for = self._handler._regex.start
                    case RegexState.END:
                        child._for = self._handler._regex.end
                    case _:
                        raise NotImplementedError()
            if not edge in self._handler._regex.edge_map[self._for,
                                                         child._for]:
                raise EdgeNotFoundError(self, edge, child._for)
            self._handler._regex.edge_map[
                self._for, child._for].remove(edge)
            child._evaluate(_taken)
        if self._handler._regex.edge_map[self._for, :].any():
            extras = {
                state: [edge for edge in (self._handler._regex
                                          .edge_map[self._for, state])]
                for state in range(self._handler._regex.size)}
            raise ExtraEdgesError(self, extras)


# functional-like interface
class assert_regex(TestCase):
    _failed_regex: MultiFigureViewer = MultiFigureViewer()

    _pattern: str
    _regex: Regex | None
    _start: NodeMatcher
    _end: NodeMatcher

    def __init__(self, pattern: str) -> None:
        self._start = NodeMatcher(self, RegexState.START)
        self._end = NodeMatcher(self, RegexState.END)
        self._pattern = pattern
        super().__init__(f"Trying to construct a regular expression "
                         f"from `{pattern}`.")

    def _call(self):
        # Initialize _regex here so errors are catched by test
        self._regex = Regex(self._pattern)
        self._callable(self._start)
        self.set_expected(
            f"Expected a DFA to be produced with "
            f"{self._start._msg([], _first=True)}")
        self._evaluate()

    def on_success(self):
        self.set_response(
            f"Produced a DFA with "
            f"{self._start._msg([], _first=True)}")

    def on_fail(self):
        fig = DebugGraphViewer(self._regex.edge_map,
                               self._regex.start,
                               self._regex.end).render()
        fig.suptitle(self._response, fontsize=8)
        fig.canvas.manager.set_window_title(
            f"{self._callable.__name__}: {self._description}")
        assert_regex._failed_regex.add(fig)

    def __exit__(self, *exc):
        return super().__exit__(*exc)

    def _evaluate(self, regex: Regex):
        self._regex = regex
        self._start._for = self._regex.start
        self._end._for = self._regex.end
        self._start._evaluate()
