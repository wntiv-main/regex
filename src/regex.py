import functools
from typing import Callable

from funcutil import extend, negate, wrap, wrap_method
from regexutil import CaptureGroup, State, Edge, MatchConditions, \
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
    
    def walk_graph(self, visitor: Callable[[Edge], None]) \
            -> Callable[['Regex'], None]:
        pass

    @wrap_method(walk_graph)
    def epsilon_closure(edge: Edge):
        # e-move -> single_input
        # single_output
        pass


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
            debug(self._start)
            if self._parse_char(self._consume_char(), _nested, debug=debug):
                break
        debug(self._start)
        for end in self._untied_ends:
            # connect to real end
            e_move = Edge()
            end.connect(e_move)
            self._end.rconnect(e_move)
            debug(self._start)
        return Regex(self._start, self._end)

    def _append_edge(self, edge: Edge):
        self._end.connect(edge)
        self._last_token = self._end
        self._end = State()
        self._end.rconnect(edge)

    def _append_regex(self, rx: Regex, group_id: CaptureGroup | None):
        # e-move for sandboxing
        sbx_edge = Edge()
        if group_id is not None:
            sbx_edge.opens.append(group_id)
        self._append_edge(sbx_edge)
        # Embed entire rx inside this
        for edge in rx._start.next.copy():
            # Steal edge from rx
            self._end.connect(edge)
        for edge in rx._start.previous.copy():
            # Steal edge from rx
            self._end.rconnect(edge)
            # pass
        self._end = State()
        # e-move for sandboxing
        sbx_edge = Edge()
        if group_id is not None:
            sbx_edge.closes.append(group_id)
        rx.end().connect(sbx_edge)
        self._end.rconnect(sbx_edge)

    def _parse_char(self,
                    char: str,
                    nested: bool,
                    debug: Callable[[State], None]):
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
                new = Edge()
                self._last_token.connect(new)
                self._end.rconnect(new)
            case '+', False:  # Repeat 1+ times quantifier
                new = Edge()
                self._last_token.rconnect(new)
                self._end.connect(new)
                # sandbox to prevend out-of-order-ing sequenced loops
                self._append_edge(Edge())
            case '*', False:  # Repeat 0+ times quantifier
                # Optional
                new = Edge()
                self._last_token.connect(new)
                self._end.rconnect(new)
                # Repeated
                new = Edge()
                self._last_token.rconnect(new)
                self._end.connect(new)
                # sandbox to prevend out-of-order-ing sequenced loops
                self._append_edge(Edge())
            case '[', False:  # Character class specifiers
                cls_specifier = self._consume_till_next(
                    ']', self._is_unescaped)
                chars = self._chars_from_char_class(cls_specifier)
                self._append_edge(Edge(functools.partial(
                    MatchConditions.try_consume, match_set=chars)))
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
                inner_group = inner_builder.build(debug=debug, _nested=True)
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
                self._append_edge(Edge(functools.partial(
                    MatchConditions.try_consume, match_char=ch)))
            case _:
                bk = '\\'  # dumb python
                raise RegexBuilder.PatternParseError(
                    f"Unexpected sequence: "
                    f"'{bk if self._escaped else ''}{char}'")
        self._escaped = False
        return False
