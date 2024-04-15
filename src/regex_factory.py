from enum import IntEnum, auto
from typing import Callable

import regex as rx
from regex_optimiser import _optimise_regex
from regexutil import CaptureGroup, ConsumeAny, ConsumeString, \
    MatchConditions, ParserPredicate, SignedSet, _parser_symbols, \
    _parser_symbols_escaped


class PatternParseError(Exception):
    message: str
    parse_src: str | None
    parse_idx: int | None

    def __init__(self, msg: str,
                 src: str | None = None,
                 idx: int | None = None) -> None:
        super().__init__(msg, src, idx)
        self.message = msg
        self.parse_src = src
        self.parse_idx = idx

    def __str__(self) -> str:
        result = self.message
        if self.parse_idx is not None and self.parse_src is not None:
            # Add parse index visual
            result += (f" at position {self.parse_idx}:"
                       f"\n\"{self.parse_src}\""
                       f"\n {' ' * self.parse_idx}^- here")
        if self.__cause__ is not None:
            cause_cls = self.__cause__.__class__.__name__
            result += f"\nCaused by {cause_cls}: {self.__cause__}"
        return result


class _NestedType(IntEnum):
    TOP = auto()
    # Bracketed groups: (a)
    NESTED_GROUP = auto()
    # Alternatives: a|b
    NESTED_ALTERNATIVE = auto()


class _RegexFactory:
    _regex: 'rx.Regex'
    _pattern: str
    _cur: int
    _last_token: 'rx.Regex | None'

    _capture_auto_id: int
    _cursor_started: int

    def __init__(self, pattern: str,
                 *, _cursor: int = 0,
                 _cid: int = 0,
                 _open_bracket_pos: int = 0) -> None:
        regex = rx.Regex(_privated=None)
        regex.edge_map = rx.Regex._empty_arr((1, 1))
        self._regex = regex
        self._pattern = pattern
        self._cur = _cursor
        # Should error if accessed before assigned
        self._last_token = None

        self._capture_auto_id = _cid
        self._cursor_started = _open_bracket_pos

    def _append(self, connection: 'ParserPredicate | rx.Regex') -> None:
        if isinstance(connection, ParserPredicate):
            connection = rx.Regex(connection, _privated=None)
        self._connect_last()
        self._last_token = connection

    def _consume_char(self) -> str:
        ch = self._pattern[self._cur]
        self._cur += 1
        return ch

    def _try_consume(self, match: str) -> bool:
        if self._pattern[self._cur:].startswith(match):
            self._cur += len(match)
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
            _started_at = self._cur
        try:
            current_attempt = self._pattern.index(ch, self._cur)
            while not predicate(current_attempt):
                current_attempt = self._pattern.index(
                    ch, current_attempt + len(ch))
            return current_attempt
        except ValueError as e:
            if ch in _open_ch_map:
                raise PatternParseError(
                    (f"Could not find closing '{ch}' for opening "
                     f"'{_open_ch_map[ch]}'"),
                    self._pattern,
                    _started_at) from e
            else:
                raise PatternParseError(
                    f"Could not find '{ch}', searching ",
                    self._pattern,
                    _started_at) from e

    def _consume_till_next(
            self,
            ch: str,
            predicate: Callable[[int], bool]):
        find_index = self._find_next(ch, predicate)
        result = self._pattern[self._cur:find_index]
        self._cur = find_index + len(ch)
        return result

    def _is_escaped(self, at: int) -> bool:
        while self._pattern[:at].endswith("\\\\"):
            at -= 2
        return self._pattern[:at].endswith("\\")

    def _is_unescaped(self, at: int) -> bool:
        return not self._is_escaped(at)

    @staticmethod
    def _chars_from_char_class(class_specifier: str) -> SignedSet[str]:
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
        result = SignedSet()
        spec_len = len(class_specifier)
        cur = 0
        while cur < spec_len:
            char: str | None = class_specifier[cur]
            if char == '-':
                raise PatternParseError("range with no start char",
                                        class_specifier, cur)
            if char == '\\':
                cur += 1
                if cur >= spec_len:
                    raise PatternParseError("Incomplete escape sequence",
                                            class_specifier, cur - 1)
                char = class_specifier[cur]
                match char:
                    case '\\' | '-' | ']':
                        pass
                    case x if (x in _parser_symbols_escaped
                               and isinstance(
                                   _parser_symbols_escaped[x],
                                   ConsumeAny)):
                        # special ranges e.g. \d, \s
                        result |= _parser_symbols_escaped[x].match_set
                        # Do not handle further
                        char = None
                    case _:
                        raise PatternParseError(
                            "Invalid escape sequence",
                            class_specifier, cur)
            if cur + 1 < spec_len and class_specifier[cur + 1] == '-':
                cur += 2
                if char is None:
                    raise PatternParseError(
                        "range did not start with character",
                        class_specifier, cur - 2)
                if cur >= spec_len:
                    raise PatternParseError("range with no end char",
                                            class_specifier, cur - 1)
                end_chr = class_specifier[cur]
                # Add set of all chars in range
                result |= SignedSet(
                    {chr(i) for i in range(ord(char),
                                           ord(end_chr) + 1)})
            elif char is not None:
                result.add(char)
            cur += 1
        return result

    def _connect_last(self):
        if self._last_token is not None:
            self._regex += self._last_token
            self._last_token = None

    def build(self, *, _nested: _NestedType = _NestedType.TOP) \
            -> 'rx.Regex':
        while self._cur < len(self._pattern):
            if self._parse_char(self._consume_char(), _nested):
                break
        self._connect_last()
        if _nested == _NestedType.TOP:
            self._regex._debug("start")
            # Loop until can match
            # self._regex.connect(self._regex.start,
            #                     self._regex.start,
            #                     MatchConditions.consume_any)
            _optimise_regex(self._regex)
        return self._regex

    def _parse_escaped(self, char: str) -> None:
        match char:
            # Special chars:
            # \A, \Z, \w, \d, etc...
            case ch if ch in _parser_symbols_escaped:
                # if ch == 'A':
                #     self._anchored = True
                self._append(_parser_symbols_escaped[ch].copy())
            case (ch, _) if ch in "\\.^$+*?[]{}()":
                self._append(ConsumeString(ch))
            case _:
                bk = '\\'  # dumb python
                raise PatternParseError(
                    f"Unexpected sequence: "
                    f"'{bk if self._escaped else ''}{char}'")

    def _parse_char(self, char: str, nested: _NestedType) -> bool:
        if char == '\\':
            self._parse_escaped(self._consume_char())
            return False
        match char:
            # ^, $, ., etc...
            case ch if ch in _parser_symbols:
                self._append(_parser_symbols[ch].copy())
            case '?':  # Make _last_token optional
                self._last_token = self._last_token.optional()
            case '+':  # Repeat 1+ times quantifier
                self._last_token = self._last_token.repeated()
                # sandbox to prevend out-of-order-ing sequenced loops
                # TODO: is this neccesary??
                # i think it is now
                # here again, thinking it still is
                self._append(MatchConditions.epsilon_transition)
            case '*':  # Repeat 0+ times quantifier
                self._last_token = self._last_token.optional().repeated()
                # see above
                self._append(MatchConditions.epsilon_transition)
            case '[':  # Character class specifiers
                negated = self._try_consume('^')
                cls_specifier = self._consume_till_next(
                    ']', self._is_unescaped)
                chars = SignedSet(
                    self._chars_from_char_class(cls_specifier),
                    negate=negated)
                self._append(ConsumeAny(chars))
            case '{':  # n-quantifier
                quantity = self._consume_till_next('}', self._is_unescaped)
                # TODO: handle ;)
                # idea:
                # for i in range(...):
                #     self._append(self._last_token)
                # idea 2: __mul__  or __imul__
                raise NotImplementedError()
            case '(':  # group
                start_pos = self._cur - 1
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
                inner_builder = _RegexFactory(
                    self._pattern,
                    _cursor=self._cur,
                    _cid=self._capture_auto_id,
                    _open_bracket_pos=start_pos)
                inner_group = inner_builder.build(
                    _nested=_NestedType.NESTED_GROUP)
                # Copy new state
                self._cur = inner_builder._cur
                self._capture_auto_id = inner_builder._capture_auto_id
                # TODO: capture groups
                # epsilon moves sandbox group and prevent loops escaping
                inner_group += MatchConditions.epsilon_transition
                self._append(inner_group)
                if nested:
                    # Ensure that we still also have closing bracket
                    _ = self._find_next(')', self._is_unescaped,
                                        _started_at=self._cursor_started)
            case ')':
                if nested != _NestedType.NESTED_GROUP:
                    # TODO: error message
                    raise PatternParseError("Unopened bracket")
                # Exit parse loop early, jump to outer group
                return True
            case '}' | ']':
                raise PatternParseError("Unopened bracket")
            case '|':  # or
                # TODO: is this safe?
                # Parse RHS of expression
                rh_builder = _RegexFactory(
                    self._pattern,
                    _cursor=self._cur,
                    _cid=self._capture_auto_id,
                    _open_bracket_pos=self._cursor_started)
                self._cur = rh_builder._cur
                self._capture_auto_id = rh_builder._capture_auto_id
                nest_type = (_NestedType.NESTED_ALTERNATIVE
                             if nested == _NestedType.TOP else nested)
                rh_group = rh_builder.build(_nested=nest_type)
                self._connect_last()
                self._regex |= rh_group
                return True
            # All other chars:
            case ch:
                self._append(ConsumeString(ch))
        return False
