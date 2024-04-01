from enum import IntFlag, auto
import functools
from typing import Callable

from funcutil import extend, negate, wrap, wrap_method
from regexutil import CaptureGroup, ConsumeAny, ConsumeString, State, Edge, MatchConditions, \
    _parser_symbols, _parser_symbols_escaped


class Regex:
    _start: State
    _end: State
    # _flags: RegexFlags
    # _capture_groups: set[CaptureGroup]

    def __init__(self, start, end) -> None:
        self._start = start
        self._end = end

    def begin(self):
        return self._start

    def end(self):
        return self._end

    class RecursionType(IntFlag):
        FORWARD = auto()
        REVERSE = auto()
        BOTH = FORWARD | REVERSE

    def walk_graph(
            self,
            visitor: Callable[[Edge], State | None],
            *args, _start: State | None = None,
            _visited: set[State] | None = None,
            _side: RecursionType = RecursionType.FORWARD,
            **kwargs) -> Callable[['Regex'], None]:
        if _start is None:
            _start = self._start
        while _start._replaced_with is not None:
            _start = _start._replaced_with
        if _visited is None:
            _visited = set()
        if _start in _visited:
            return
        _visited.add(_start)
        for edge in _start.next.copy():
            if edge.next is None or edge.previous is None:
                continue
            if _side & Regex.RecursionType.FORWARD:
                visitor(
                    edge, *args, **kwargs,
                    _start=self._start, _end=self._end)
            if edge.next is not None:
                self.walk_graph(visitor, *args,
                                **kwargs,
                                _start=edge.next,
                                _visited=_visited)
            else:
                _visited.discard(_start)
                self.walk_graph(visitor, *args,
                                **kwargs,
                                _start=_start,
                                _visited=_visited)
                return
            if _side & Regex.RecursionType.REVERSE:
                visitor(
                    edge, *args, **kwargs,
                    _start=self._start, _end=self._end)

    @wrap_method(walk_graph)
    def epsilon_closure(
            edge: Edge,
            _start: State,
            _end: State,
            debug=lambda _: None):
        if (edge.is_free() and edge.previous == edge.next):
            next_state = edge.previous
            with edge:
                edge.remove()
            debug(_start, _end, f"remove self-loop from {next_state}")
            return
        # transfer capture groups to non-epsilon transitions
        if (edge.predicate == MatchConditions.epsilon_transition
                and (edge.has_opens() or edge.has_closes())):
            with edge:
                if edge.has_opens() and edge.next.inputs() == 1:
                    try:  # spoof with statement dont question it
                        for path in edge.next.next:
                            path.__enter__()
                        for group in edge.move_opens():
                            for path in edge.next.next:
                                path.open(group)
                    finally:
                        for path in edge.next.next:
                            path.__exit__(*(None,)*3)
                if edge.has_closes() and edge.previous.outputs() == 1:
                    try:  # spoof with statement dont question it
                        for path in edge.previous.previous:
                            path.__enter__()
                        for group in edge.move_closes():
                            for path in edge.previous.previous:
                                path.close(group)
                    finally:
                        for path in edge.previous.previous:
                            path.__exit__(*(None,)*3)
        # merge states connected by e-moves
        if (edge.is_free()
            and (edge.previous.outputs() == 1
                 or edge.next.inputs() == 1)):
            debug_str = f"{edge}: merge {edge.next} with {edge.previous}"
            edge.next.merge(edge.previous)
            debug(_start, _end, debug_str)
            return

    @wrap_method(walk_graph, _side=RecursionType.REVERSE)
    def extended_epsilon_closure(
            edge: Edge,
            _start: State,
            _end: State,
            debug=lambda _: None):
        if (edge.is_free() and edge.previous == edge.next):
            next_state = edge.previous
            with edge:
                edge.remove()
            debug(_start, _end, f"remove self-loop from {next_state}")
            return
        # strategy for removing enclosed e-moves: split their end-state
        # into 2 - one for the e-move, one for the other connections
        if (edge.is_free()
                and edge.previous.outputs() > 1
                and edge.next.inputs() > 1):
            debug(_start, _end, f"{edge}: spliting {edge.next}")
            new_state = edge.next.clone_shallow(reverse=False)
            edge.previous.merge(new_state)
            if edge.next == _end:
                with Edge() as new_edge:
                    new_edge.previous = edge.previous
                    new_edge.next = _end
            with edge:
                edge.remove()
            debug(_start, _end, f"{edge}: split {edge.next} to {new_state}")

    @wrap_method(walk_graph)
    def powerset_construction(
            edge: Edge,
            _start: State,
            _end: State):
        for other in edge.previous.next.copy():
            if edge.next == other.next:
                continue
            match edge.predicate_intersection(other):
                case None: continue
                case left, intersect, right:
                    # We need to construct a "superposition" state
                    start_state = edge.previous
                    new_state = edge.next.clone_shallow(reverse=False)
                    new_state.merge(other.next.clone_shallow(reverse=False))
                    with edge:
                        if left is not None:
                            edge.predicate = left
                        else:
                            edge.remove(chain=True)
                    with other:
                        if right is not None:
                            other.predicate = right
                        else:
                            other.remove(chain=True)
                    with Edge(intersect) as intersect_edge:
                        intersect_edge.previous = start_state
                        intersect_edge.next = new_state


    def minify(self):
        states: set[State] = set()
        to_explore: list[State] = [self._start]
        while to_explore:
            exploring = to_explore.pop()
            if exploring in states:
                continue
            states.add(exploring)
            for edge in exploring.next:
                if edge.next == exploring:
                    continue
                to_explore.append(edge.next)
        states_list = list(states)
        for i in range(len(states_list) - 1):  # exclude last
            first = states_list[i]
            # only iterate forward states
            for second in states_list[i + 1:]:
                diff = first.output_diff(second)
                for edge in diff:
                    # TODO: this is wrong. fix
                    if not edge.is_free() or (edge.next != first
                                              and edge.next != second):
                        break
                else:
                    first.merge(second)



class RegexBuilder:
    class PatternParseError(Exception):
        message: str

        def __init__(self, msg: str, e: Exception | None = None) -> None:
            super().__init__(msg, e)
            self.message = msg
            self.__cause__ = e

        def __str__(self) -> str:
            return f"{self.message}Caused by "\
                f"{self.__cause__.__class__.__name__}: {self.__cause__}"

    _special_chars = "\\.^$+*?[]{}()"

    _pattern: str
    _cursor: int
    _cursor_started: int

    _escaped: bool
    _start: State
    _last_token: State
    _end: State
    _untied_ends: set[State]

    _capture_auto_id: int

    def __init__(self, pattern: str,
                 *, _cursor: int = 0,
                 _open_bracket_pos: int = 0,
                 _cid: int = 0):
        self._pattern = pattern
        self._cursor = _cursor
        self._cursor_started = _open_bracket_pos

        self._escaped = False
        self._start = self._last_token = self._end = State()
        self._untied_ends = set()

        self._capture_auto_id = _cid

    def _consume_char(self) -> str:
        ch = self._pattern[self._cursor]
        self._cursor += 1
        return ch

    def _try_consume(self, match: str) -> bool:
        if self._pattern[self._cursor:].startswith(match):
            self._cursor += len(match)
            return True
        return False

    def _find_next(
            self,
            ch: str,
            predicate: Callable[[int], bool],
            *, _started_at: int | None = None,
            _open_ch_map={  # Static opject, initialized only once
                ']': '[',
                '}': '{',
                ')': '(',
                '>': '<'
            }):
        if _started_at is None:
            _started_at = self._cursor
        try:
            current_attempt = self._pattern.index(ch, self._cursor)
            while not predicate(current_attempt):
                current_attempt = self._pattern.index(
                    ch, current_attempt + len(ch))
            return current_attempt
        except ValueError as e:
            if ch in _open_ch_map:
                raise RegexBuilder.PatternParseError(
                    f"""Could not find closing '{ch}' for opening \
'{_open_ch_map[ch]}' at position {_started_at}:
"{self._pattern}"
 {' ' * _started_at}^ here
""", e)
            else:
                raise RegexBuilder.PatternParseError(
                    f"""Could not find '{ch}' searching from position \
{_started_at}:
"{self._pattern}"
 {' ' * _started_at}^ here
""", e)

    def _consume_till_next(
            self,
            ch: str,
            predicate: Callable[[int], bool]):
        find_index = self._find_next(ch, predicate)
        result = self._pattern[self._cursor:find_index]
        self._cursor = find_index + len(ch)
        return result

    def _is_escaped(self, at: int):
        while self._pattern[:at].endswith("\\\\"):
            at -= 2
        return self._pattern[:at].endswith("\\")

    @negate
    @extend(_is_escaped)
    def _is_unescaped(self, at: int):
        pass

    @staticmethod
    def _chars_from_char_class(class_specifier: str) -> set[str]:
        """
        Produces the set of all chars that satisfy the given regular
        expression char. class specifier (see
        https://www.rexegg.com/regex-quickstart.html#classes).

        Arguments:
            class_specifier -- the string representation of the char
                class specifier, excluding the square brackets. e.g
                "A-Z", "0-9a-z_", etc.

        Returns:
            A set of all the conforming characters.
        """
        # TODO: handle ;)
        # ranges:
        # {chr(i) for i in range(ord(start), ord(end) + 1)}

    def build(self,
              *, debug: Callable[[State], None] = lambda _: None,
              _nested: bool = False) -> Regex:
        while self._cursor < len(self._pattern):
            if self._parse_char(self._consume_char(), _nested):
                break
        for end in self._untied_ends:
            # connect to real end
            with Edge() as e_move:
                e_move.previous = end
                e_move.next = self._end
        result = Regex(self._start, self._end)
        if not _nested:
            # pass
            result.epsilon_closure(debug=debug)
            result.epsilon_closure(debug=debug)
            result.extended_epsilon_closure(debug=debug)
            result.extended_epsilon_closure(debug=debug)
            result.extended_epsilon_closure(debug=debug)
            result.epsilon_closure(debug=debug)
            result.epsilon_closure(debug=debug)
            result.extended_epsilon_closure(debug=debug)
            result.extended_epsilon_closure(debug=debug)
            result.extended_epsilon_closure(debug=debug)
            result.epsilon_closure(debug=debug)
            result.epsilon_closure(debug=debug)
            result.epsilon_closure(debug=debug)
            result.minify()
            result.epsilon_closure(debug=debug)
            result.epsilon_closure(debug=debug)
            result.epsilon_closure(debug=debug)
            result.powerset_construction()
        return result

    def _append_edge(self, edge: Edge):
        with edge:
            edge.previous = self._end
            self._last_token = self._end
            self._end = State()
            edge.next = self._end

    def _append_regex(self, rx: Regex, group_id: CaptureGroup | None):
        # e-moves for sandboxing
        with Edge() as start_edge, Edge() as end_edge:
            if group_id is not None:
                start_edge.open(group_id)
                end_edge.close(group_id)

            start_edge.previous = self._end
            start_edge.next = rx.begin()
            end_edge.previous = rx.end()
            self._end = State()
            end_edge.next = self._end

    def _parse_char(self,
                    char: str,
                    nested: bool):
        match char, self._escaped:
            case '\\', False:
                self._escaped = True
                # early return; dont reset `escaped` flag
                return False
            # Special chars:
            # \A, \Z, \w, \d, etc...
            case ch, True if ch in _parser_symbols_escaped:
                self._append_edge(Edge(_parser_symbols_escaped[ch]))
            # ^, $, ., etc...
            case ch, False if ch in _parser_symbols:
                self._append_edge(Edge(_parser_symbols[ch]))
            case '?', False:  # Make _last_token optional
                with Edge() as edge:
                    edge.previous = self._last_token
                    edge.next = self._end
            case '+', False:  # Repeat 1+ times quantifier
                with Edge() as edge:
                    edge.previous = self._end
                    edge.next = self._last_token
                # sandbox to prevend out-of-order-ing sequenced loops
                self._append_edge(Edge())
            case '*', False:  # Repeat 0+ times quantifier
                # # Optional
                # new = Edge()
                # self._last_token.connect(new)
                # self._end.rconnect(new)
                # # Repeated
                # new = Edge()
                # self._last_token.rconnect(new)
                # self._end.connect(new)
                # Start and end of loop are same state
                self._last_token.merge(self._end)
                self._end = self._last_token
                # sandbox to prevend out-of-order-ing sequenced loops
                self._append_edge(Edge())
            case '[', False:  # Character class specifiers
                cls_specifier = self._consume_till_next(
                    ']', self._is_unescaped)
                chars = self._chars_from_char_class(cls_specifier)
                self._append_edge(Edge(ConsumeAny(chars)))
            case '{', False:  # n-quantifier
                quantity = self._consume_till_next('}', self._is_unescaped)
                # TODO: handle ;)
            case '(', False:  # group
                start_pos = self._cursor - 1
                # Capture group time!
                capture_group: CaptureGroup | None = None
                if self._try_consume("?:"):
                    # Non-capturing group
                    pass
                elif self._try_consume("?<") or self._try_consume("?P<"):
                    # Named capture group
                    capture_group = self._consume_till_next(
                        '>', lambda _: True)
                else:
                    # Capturing group
                    self._capture_auto_id += 1
                    capture_group = self._capture_auto_id
                # Ensure that capture group will have closing bracket
                _ = self._find_next(')', self._is_unescaped)
                inner_builder = RegexBuilder(
                    self._pattern,
                    _cursor=self._cursor,
                    _open_bracket_pos=start_pos,
                    _cid=self._capture_auto_id)
                inner_group = inner_builder.build(_nested=True)
                # Copy new state
                self._cursor = inner_builder._cursor
                self._capture_auto_id = inner_builder._capture_auto_id
                self._append_regex(inner_group, capture_group)
                if nested:
                    # Ensure that we still also have closing bracket
                    _ = self._find_next(')', self._is_unescaped,
                                        _started_at=self._cursor_started)
            case ')', False if nested:
                # Exit parse loop early, jump to outer group
                return True
            case '|', False:  # or
                # Two options:
                # _start -> _end
                # _end -> [end of parse]
                # Connect first option to end
                self._untied_ends.add(self._end)
                # Start second option at _start,
                # continue parsing from there
                self._end = self._start
            # All other chars:
            case (ch, _) if not self._escaped or ch in self._special_chars:
                self._append_edge(Edge(ConsumeString(ch)))
            case _:
                bk = '\\'  # dumb python
                raise RegexBuilder.PatternParseError(
                    f"Unexpected sequence: "
                    f"'{bk if self._escaped else ''}{char}'")
        self._escaped = False
        return False
