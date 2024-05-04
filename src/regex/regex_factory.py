"""Utilities for constructing and parsing a Regex"""

__author__ = "Callum Hynes"
__all__ = ['PatternParseError', '_RegexFactory']

from enum import IntEnum, auto
from typing import Callable

from .regex import Regex
from .regex_optimiser import _OptimiseRegex
from .regexutil import (CaptureGroup, ConsumeAny, ConsumeString,
                        MatchConditions, ParserPredicate, SignedSet,
                        _parser_symbols, _parser_symbols_escaped)


class PatternParseError(Exception):
    """
    An error which occured during the parsing of the regular expression
    pattern.
    """
    message: str
    """The human-readable message associated with the error."""
    parse_src: str | None
    """The pattern which caused the error."""
    parse_idx: int | None
    """The index into the pattern where the error was caused."""

    def __init__(self, msg: str,
                 src: str | None = None,
                 idx: int | None = None) -> None:
        """
        Initializes a regular expression parser error.

        Arguments:
            msg -- The human-readable message associated with the error
            src -- The pattern in which the error was found
                (default: {None})
            idx -- The index within the pattern where the error was
                found (default: {None})
        """
        super().__init__(msg, src, idx)
        self.message = msg
        self.parse_src = src
        self.parse_idx = idx

    def __str__(self) -> str:
        """
        Human-readable string representaion of the error.
        
        Returns:
            human-readable error message, with indication of where
            the error occured within the source string.
        """
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
    """
    The way that _RegexFactory was called, which has slight impacts on
    it's functionality.
    """

    TOP = auto()
    """
    Top-level; no nesting. Closing paranthesis cause a parse error.
    """

    NESTED_GROUP = auto()
    """
    Nested call to _RegexFactory.build by a bracketed group. Closing
    paranthesis close the group.
    """

    NESTED_ALTERNATIVE = auto()
    """
    Nested call to _RegexFactory.build by a union (|) operator. Closing
    paranthesis delegate handling to the caller.
    """


class _RegexFactory:
    """Factory class to parse and construct a Regex"""

    regex: Regex
    """The Regex being currently built"""

    pattern: str
    """The string pattern being parsed"""

    _cur: int
    """The current parse index into {_pattern}"""

    _last_token: Regex | None
    """The last parsed token, stored seperately for `?`, `+`, etc."""

    _capture_auto_id: int
    """Auto-incrementing ID for capture groups"""

    _cursor_started: int
    """The index into {_pattern} where this factory started parsing"""

    def __init__(self, pattern: str,
                 *, _cursor: int = 0,
                 _cid: int = 0,
                 _open_bracket_pos: int = 0) -> None:
        """
        Initialize parser for the given pattern, starting at a given
        position ({_cursor)}.

        Arguments:
            pattern -- The regular expression as a string, to parse.

        Keyword Arguments:
            _cursor -- The index into the pattern to start parsing
                (default: {0})
            _cid -- Initial value for the auto-incrementing ID
                (default: {0})
            _open_bracket_pos -- Position of the opening bracket, if
                any, used for detailed error printouts. (default: {0})
        """
        regex = Regex(_privated=None)
        self.regex = regex
        self.pattern = pattern
        self._cur = _cursor
        # Should error if accessed before assigned
        self._last_token = None

        self._capture_auto_id = _cid
        self._cursor_started = _open_bracket_pos

    def append(self, connection: ParserPredicate | Regex) -> None:
        """
        Append to the regex, correctly handling tokens for {_last_token}

        Arguments:
            connection -- The connection or regular expression to append
        """
        if isinstance(connection, ParserPredicate):
            connection = Regex(connection, _privated=None)
        self.connect_last()
        self._last_token = connection

    def _consume_char(self) -> str:
        """
        Consume a character from the pattern

        Returns:
            The character that was consumed
        """
        ch = self.pattern[self._cur]
        self._cur += 1
        return ch

    def _try_consume(self, match: str) -> bool:
        """
        Try to consume the given string from the front of the pattern,
        if the string matches.

        Arguments:
            match -- The string to try consume

        Returns:
            Whether or not the string was found and consumed
        """
        if self.pattern[self._cur:].startswith(match):
            self._cur += len(match)
            return True
        return False

    # Dangerous default value intentionally used to avoid
    # re-initialization of a constant value ("static" initialization),
    # for performance
    # pylint: disable-next=dangerous-default-value
    def _find_next(
            self,
            ch: str,
            predicate: Callable[[int], bool] = lambda _: True,
            *, _started_at: int | None = None,
            _OPEN_CH_MAP={
                ']': '[',
                '}': '{',
                ')': '(',
                '>': '<'
            }) -> int:
        """
        Find the next occurance of the given character in the pattern

        Arguments:
            ch -- The string to search for
            predicate -- A predicate for whether the search should stop,
                passed the current index; search stops if predicate
                returns True

        Keyword Arguments:
            _started_at -- The start position of the search, used in
                detailed error messages (default: {None})

        Raises:
            PatternParseError: If the substring is not found in the
                pattern

        Returns:
            The index of the found string
        """
        if _started_at is None:
            _started_at = self._cur - 1
        try:
            current_attempt = self.pattern.index(ch, self._cur)
            while not predicate(current_attempt):
                current_attempt = self.pattern.index(
                    ch, current_attempt + len(ch))
            return current_attempt
        except ValueError as e:
            if ch in _OPEN_CH_MAP:
                raise PatternParseError(
                    (f"Could not find closing '{ch}' for opening "
                     f"'{_OPEN_CH_MAP[ch]}'"),
                    self.pattern,
                    _started_at) from e
            raise PatternParseError(
                f"Could not find '{ch}', searching ",
                self.pattern,
                _started_at) from e

    def _consume_till_next(
            self,
            ch: str,
            predicate: Callable[[int], bool] = lambda _: True):
        """
        Find the next occurance of the given character in the pattern,
        and consume up until (and including) that point

        Arguments:
            ch -- The string to search for
            predicate -- A predicate for whether the search should stop,
                passed the current index; search stops if predicate
                returns True

        Raises:
            PatternParseError: If the substring is not found in the
                pattern

        Returns:
            Everything that was consumed, until (and excluding) the
            search string
        """
        find_index = self._find_next(ch, predicate)
        result = self.pattern[self._cur:find_index]
        self._cur = find_index + len(ch)
        return result

    def _is_escaped(self, at: int) -> bool:
        """
        Finds whether the given char is escaped by backslashes. This
        requires that it is preceded by an odd number of backslashes

        Arguments:
            at -- The index to search

        Returns:
            Whether the char is escaped
        """
        while self.pattern[:at].endswith("\\\\"):
            at -= 2
        return self.pattern[:at].endswith("\\")

    def _is_unescaped(self, at: int) -> bool:
        """
        Finds whether the given char is not escaped by backslashes. This
        requires that it is preceded by an even number of backslashes

        Arguments:
            at -- The index to search

        Returns:
            Whether the char is not escaped
        """
        return not self._is_escaped(at)

    @staticmethod
    def chars_from_char_class(class_specifier: str) -> SignedSet[str]:
        """
        Produces the set of all chars that satisfy the given regular
        expression char. class specifier (see
        https://www.rexegg.com/regex-quickstart.html#classes).

        Arguments:
            class_specifier -- the string representation of the char
                class specifier, excluding the square brackets. e.g
                "A-Z", "0-9a-z_", etc.

        Returns:
            A SignedSet of all the conforming characters.
        """
        result = SignedSet()
        spec_len = len(class_specifier)
        cur = 0
        while cur < spec_len:
            char: str | None = class_specifier[cur]
            if char == '-':
                raise PatternParseError(
                    "range with no start char", class_specifier, cur)
            # Handle escapes
            if char == '\\':
                cur += 1
                if cur >= spec_len:
                    raise PatternParseError(
                        "Incomplete escape sequence",
                        class_specifier, cur - 1)
                char = class_specifier[cur]
                match char:
                    #        Hello!     Hey.
                    #           \       /
                    case '\\' | '-' | '^' | ']':
                        pass
                    case x if (x in _parser_symbols_escaped
                               and isinstance(
                                   _parser_symbols_escaped[x],
                                   ConsumeAny)):
                        # special ranges e.g. \d, \s
                        # not sure this is standard but its useful so...
                        result |= (_parser_symbols_escaped[x]
                                   .match_set)  # type: ignore
                        # Do not handle further
                        char = None
                    case _:
                        raise PatternParseError(
                            "Invalid escape sequence",
                            class_specifier, cur)
            # Handle char ranges
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
            # Normal chars
            elif char is not None:
                result.add(char)
            cur += 1
        return result

    def connect_last(self):
        """Connect the {_last_token} to the Regex"""
        if self._last_token is not None:
            self.regex += self._last_token
            self._last_token = None

    def _require_previous(self):
        """Raise a parse exception if {_last_token} was not set"""
        if self._last_token is None:
            raise PatternParseError(
                f"'{self.pattern[self._cur - 1]}' must be preceded by"
                f" another token", self.pattern, self._cur - 1)

    def build(self, *, _nested: _NestedType = _NestedType.TOP) -> Regex:
        """
        Parse the pattern and return the resultant Regex

        Returns:
            The final Regex produced
        """
        while self._cur < len(self.pattern):
            if self.parse_char(self._consume_char(), _nested):
                break
        self.connect_last()
        if _nested == _NestedType.TOP:
            # Only do optimisation on top-level
            # pylint: disable=protected-access
            self.regex._debug("start")
            # Loop until can match
            _OptimiseRegex(self.regex)
            self.regex._base = self.regex.copy()
            self.regex._prepare_for_use()
            # Dont lazily initialize reverse if using Regex constructor
            self.regex._prepare_full_reverse()
        return self.regex

    def parse_escaped(self, char: str) -> None:
        """
        Handle parsing of escaped chars in the pattern

        Arguments:
            char -- The char to handle

        Raises:
            PatternParseError: If a char is escaped that shouldn't be
        """
        match char:
            # Special chars:
            # \A, \Z, \w, \d, etc...
            case ch if ch in _parser_symbols_escaped:
                # if ch == 'A':
                #     self._anchored = True
                self.append(_parser_symbols_escaped[ch].copy())
            case ch if ch in "\\.^$+*?[]{}()":
                self.append(ConsumeString(ch))
            case _:
                raise PatternParseError(
                    f"Unexpected sequence: "
                    f"'\\{char}'")

    # pylint: disable-next=design
    def parse_char(self, char: str, nested: _NestedType) -> bool:
        """
        Handle parsing of chars in the pattern

        Arguments:
            char -- The char to parse
            nested -- The nesting state of the parser

        Raises:
            PatternParseError: If there is any unexpected error while
                parsing

        Returns:
            Whether the parse loop should break
        """
        if char == '\\':
            self.parse_escaped(self._consume_char())
            return False
        match char:
            # ^, $, ., etc...
            case ch if ch in _parser_symbols:
                self.append(_parser_symbols[ch].copy())
            case '?':  # Make _last_token optional
                self._require_previous()
                assert self._last_token is not None
                self._last_token.optional()
            case '+':  # Repeat 1+ times quantifier
                self._require_previous()
                assert self._last_token is not None
                self._last_token.repeat()
            case '*':  # Repeat 0+ times quantifier
                self._require_previous()
                assert self._last_token is not None
                self._last_token.optional().repeat()
            case '[':  # Character class specifiers
                negated = self._try_consume('^')
                start_cur = self._cur
                cls_specifier = self._consume_till_next(
                    ']', self._is_unescaped)
                try:
                    chars = self.chars_from_char_class(cls_specifier)
                except PatternParseError as e:
                    # Re-raise with context
                    e.parse_src = self.pattern
                    if e.parse_idx is None:
                        e.parse_idx = 0
                    e.parse_idx += start_cur
                    raise e
                if negated:
                    chars.negate()
                self.append(ConsumeAny(chars))
            case '{':  # n-quantifier
                self._require_previous()
                assert self._last_token is not None
                start_cur = self._cur
                quantity_str = self._consume_till_next(
                    '}', self._is_unescaped)

                def parse_int(x: str) -> int | None:
                    x = x.strip()
                    if not x:  # null value
                        return None
                    if not x.isnumeric():
                        raise PatternParseError(
                            "Invalid n-quantifier numeral",
                            self.pattern,
                            start_cur + quantity_str.index(x.strip()))
                    return int(x)
                quantity = tuple(map(
                    parse_int,
                    quantity_str.split(',')))
                if len(quantity) > 2:
                    raise PatternParseError(
                        "Invalid n-quantifier: should have no more than"
                        " two values",
                        self.pattern,
                        # Index of second comma
                        start_cur + quantity_str.index(
                            ',', quantity_str.index(',') + 1))
                # Last value must be > 0
                if quantity[-1] == 0:
                    raise PatternParseError(
                        "Invalid n-quantifier: max must be > 0",
                        self.pattern, self._cur - 2)
                # epsilon moves sandbox group and prevent loops escaping
                self._last_token += MatchConditions.epsilon_transition
                match quantity:
                    case (int(n),):
                        self._last_token *= n
                    case (int(n), int(m)):
                        if m > n:
                            self._last_token = (
                                self._last_token * n
                                + self._last_token.optional() * (m - n))
                        elif m == n:
                            self._last_token *= n
                        else:
                            raise PatternParseError(
                                "Invalid n-quantifier: max is smaller"
                                " than min",
                                self.pattern, start_cur)
                    case (None | 0, int(m)):
                        self._last_token = (
                            self._last_token.optional() * m)
                    case (int(n), None):
                        self._last_token = (
                            self._last_token * n
                            + self._last_token.optional().repeat())
                    case _:
                        raise PatternParseError(
                            "Invalid n-quantifier syntax",
                            self.pattern, start_cur)
            case '(':  # group
                start_pos = self._cur - 1
                # Capture groups, currently ignored
                # Are available here for future extension
                # pylint: disable-next=unused-variable
                capture_group: CaptureGroup | None = None
                if self._try_consume("?:"):
                    # Non-capturing group
                    pass
                elif (self._try_consume("?<")
                      or self._try_consume("?P<")):
                    # Named capture group
                    capture_group = self._consume_till_next('>')
                else:
                    # Capturing group
                    self._capture_auto_id += 1
                    capture_group = self._capture_auto_id
                # Ensure that capture group will have closing bracket
                _ = self._find_next(')', self._is_unescaped)
                inner_builder = _RegexFactory(
                    self.pattern,
                    _cursor=self._cur,
                    _cid=self._capture_auto_id,
                    _open_bracket_pos=start_pos)
                inner_group = inner_builder.build(
                    _nested=_NestedType.NESTED_GROUP)
                # Copy new state
                # pylint: disable=protected-access
                self._cur = inner_builder._cur
                self._capture_auto_id = inner_builder._capture_auto_id
                # epsilon moves sandbox group and prevent loops escaping
                inner_group += MatchConditions.epsilon_transition
                self.append(inner_group)
                if nested == _NestedType.NESTED_GROUP:
                    # Ensure that we still also have closing bracket
                    _ = self._find_next(
                        ')', self._is_unescaped,
                        _started_at=self._cursor_started)
            case ')':
                if nested == _NestedType.TOP:
                    raise PatternParseError(
                        f"Unopened '{char}'",
                        self.pattern, self._cur - 1)
                if nested == _NestedType.NESTED_ALTERNATIVE:
                    self._cur -= 1  # Do not consume bracket
                # Exit parse loop early, jump to outer group
                return True
            case '}' | ']':
                raise PatternParseError(
                    f"Unopened '{char}'",
                    self.pattern, self._cur - 1)
            case '|':  # or
                self._require_previous()
                start_cur = self._cur - 1
                # Parse RHS of expression
                rh_builder = _RegexFactory(
                    self.pattern,
                    _cursor=self._cur,
                    _cid=self._capture_auto_id,
                    _open_bracket_pos=self._cursor_started)
                # pylint: disable=protected-access
                rh_group = rh_builder.build(
                    _nested=_NestedType.NESTED_ALTERNATIVE)
                self._cur = rh_builder._cur
                self._capture_auto_id = rh_builder._capture_auto_id
                self.connect_last()
                # Make sure to not double-add
                self._last_token = None
                if not rh_group:
                    raise PatternParseError(
                        "'|' must be succeeded by atleast one token",
                        self.pattern, start_cur)
                self.regex |= rh_group
            # All other chars:
            case ch:
                self.append(ConsumeString(ch))
        return False
