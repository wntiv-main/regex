from enum import IntEnum, auto
from typing import Callable, Self

from src.debug_graph_viewer import DebugGraphViewer, MultiFigureViewer
from src import Regex
from src.regex_factory import PatternParseError
from src.regexutil import ConsumeAny, ConsumeString, ParserPredicate, SignedSet, State
from .test import AssertRaises, ResultType, TestCase, TestType
from .test_error import EdgeNotFoundError, ExtraEdgesError, RegexMatchError, StateIdentityError


class RegexState(IntEnum):
    START = auto()
    SELF = auto()
    END = auto()
    ANY = auto()


class NodeMatcher:
    _handler: 'TestRegexShape'
    _children: list[tuple[ParserPredicate, 'NodeMatcher']]
    _for: State | None
    _evaluated: bool
    _type: RegexState
    _expected_type: RegexState

    def __init__(self,
                 handler: 'TestRegexShape',
                 state_type: RegexState = RegexState.ANY):
        self._handler = handler
        self._children = []
        self._for = None
        self._evaluated = False
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

    def has_literal_chain(self, literal_chain: str,
                          to: 'NodeMatcher | RegexState'
                          = RegexState.ANY) -> 'NodeMatcher':
        last = self
        # Skip last in loop to pass `to` arg
        for char in literal_chain[:-1]:
            last = last.has_literal(char)
        return last.has_literal(literal_chain[-1], to)

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
        return next

    @staticmethod
    def _num_ending(num: int, *,
                    _endings=['th', 'st', 'nd', 'rd']) -> str:
        if (num >= 10 and num < 20) or num % 10 > 3:
            return 'th'
        else:
            return _endings[num % 10]

    @staticmethod
    def _num_w_ending(num: int) -> str:
        return f"{num}{NodeMatcher._num_ending(num)}"

    def _msg(self,
             _visited: list['NodeMatcher'],
             _indent: int = 0,
             _top: bool = False,
             *, _TAB="\t") -> str:
        if self in _visited:
            if self._type == RegexState.END:
                return "to the end state."
            if self._type == RegexState.START:
                return "to the start state."
            left_idx = _visited.index(self)
            right_idx = len(_visited) - left_idx
            relative_pos: str
            if right_idx == 1:
                relative_pos = "current"
            elif right_idx == 2:
                relative_pos = "previous"
            elif left_idx == 0:
                relative_pos = "start"
            elif left_idx > right_idx:
                relative_pos = (NodeMatcher._num_w_ending(right_idx - 1)
                                + " previous")
            else:
                relative_pos = NodeMatcher._num_w_ending(left_idx + 1)
            return f"back to the {relative_pos} state."
        _visited.append(self)
        if len(self._children) == 0:
            joiner = ""
            if not _top:
                joiner = " to "
            return f"{joiner}{self._state_name()}."
        elif len(self._children) == 1:
            result = ""
            if _top or self._type != RegexState.ANY:
                result = self._state_name()
            result += f", followed by an {self._children[0][0]}-move"
            result += self._children[0][1]._msg(_visited, _indent)
            return result
        else:
            result = ""
            if not _top:
                result = " to "
            result += f"{self._state_name()}, followed by:<ul>"
            for move, child in self._children:
                result += (f"<li>an {move}-move to "
                           f"{child._msg(_visited, _indent + 1, True)}"
                           f"</li>")
            result += "</ul>"
            return result

    def _evaluate(self, _taken: set[State]):
        if self._evaluated:
            return  # Already been visited
        self._evaluated = True
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
                            if (state in _taken
                                or state == self._handler._regex.start
                                    or state == self._handler._regex.end):
                                # State already has NodeMatcher
                                continue
                            if edge.kind_of_in(self._handler._regex
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
            # cursedness since ParserPredicates are mutable
            found_edge = edge.kind_of_in(self._handler._regex.edge_map[
                self._for, child._for])
            if found_edge is None:
                # Connection not found
                raise EdgeNotFoundError(self, edge, child._for)
            self._handler._regex.edge_map[
                self._for, child._for].remove(found_edge)
            child._evaluate(_taken)
        if self._handler._regex.edge_map[self._for, :].any():
            extras = {
                state: [edge for edge in (self._handler._regex
                                          .edge_map[self._for, state])]
                for state in range(self._handler._regex.size)}
            raise ExtraEdgesError(self, extras)


# functional-like interface
class TestRegexShape(TestCase):
    _failed_regex: MultiFigureViewer = MultiFigureViewer()

    _pattern: str
    _regex: Regex | None
    _debug_regex: Regex | None
    _start: NodeMatcher
    _end: NodeMatcher
    _expecting: str | None
    _callable: Callable[[NodeMatcher], None]

    def __init__(self, pattern: str,
                 test_type: TestType = TestType.EXPECTED,
                 expecting: str | None = None):
        super().__init__(
            test_type,
            f"Trying to construct regular expression from `{pattern}`")
        self._start = NodeMatcher(self, RegexState.START)
        self._end = NodeMatcher(self, RegexState.END)
        self._pattern = pattern
        self._expecting = expecting

    # For use as decorator
    def __call__(self, func: Callable[[NodeMatcher], None]) -> None:
        self._callable = func
        self._description = (f"{func.__name__.replace('_', ' ')}: "
                             f"{self._description}")

    def _inner_test(self):
        # Initialize _regex here so errors are catched by test
        self._regex = Regex(self._pattern)
        self._debug_regex = self._regex.copy()
        self._callable(self._start)
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

    def run(self):
        super().run()
        if self._result == ResultType.FAIL:
            fig = DebugGraphViewer(self._debug_regex.edge_map,
                                   self._debug_regex.start,
                                   self._debug_regex.end).render()
            fig.suptitle(self._outcome, fontsize=8)
            fig.canvas.manager.set_window_title(
                f"{self._callable.__name__}: {self._description}")
            TestRegexShape._failed_regex.add(fig)

    def _evaluate(self):
        self._start._for = self._regex.start
        self._end._for = self._regex.end
        self._start._evaluate(set())


class TestRegexMatches(TestCase):
    _pattern: str
    _regex: Regex | None
    _expected_matches: set[str]
    _unexpected_matches: set[str]

    def __init__(self, pattern: str,
                 test_type: TestType = TestType.EXPECTED):
        super().__init__(test_type, f"Testing matches for `{pattern}`")
        self._pattern = pattern
        # Defer initialization for error capture
        self._regex = None
        self._expected_matches = set()
        self._unexpected_matches = set()

    def _inner_test(self) -> None:
        self.set_expected(f"Expected {self._expected_matches} to all "
                          f"match, and {self._unexpected_matches} to "
                          f"all not match.")
        self._regex = Regex(self._pattern)
        for test in self._expected_matches:
            if not self._regex.is_in(test):
                raise RegexMatchError(self._regex, test, True)
        for test in self._unexpected_matches:
            if self._regex.is_in(test):
                raise RegexMatchError(self._regex, test, False)

    def assert_matches(self, *tests: str) -> Self:
        for test in tests:
            self._expected_matches.add(test)
        return self

    def assert_doesnt_match(self, *tests: str) -> Self:
        for test in tests:
            self._unexpected_matches.add(test)
        return self


class TestNoParseError(TestCase):
    _pattern: str

    def __init__(self, pattern: str,
                 test_type: TestType = TestType.INVALID):
        super().__init__(
            test_type,
            f"Building regular expression from valid pattern "
            f"`{pattern}`",
            expected=f"Pattern to be parsed with no exceptions.")
        self._pattern = pattern
        self._callable = Regex

    def _inner_test(self) -> None:
        Regex(self._pattern)


class TestParseError(AssertRaises):
    _pattern: str

    def __init__(self, pattern: str,
                 test_type: TestType = TestType.INVALID):
        super().__init__(
            PatternParseError,
            f"Building regular expression from invalid pattern "
            f"`{pattern}`",
            test_type)
        self._pattern = pattern
        self._callable = Regex

    def _inner_test(self) -> None:
        return super()._inner_test(self._pattern)
