__all__ = ["PatternParseError", "Regex", "State"]
__author__ = "Callum Hynes"

from typing import Any, Callable, Self, TypeAlias, overload
import numpy as np

from regexutil import CaptureGroup, ConsumeAny, ConsumeString, \
    MatchConditions, ParserPredicate, SignedSet, _parser_symbols, \
    _parser_symbols_escaped


State: TypeAlias = int


class PatternParseError(Exception):
    message: str

    def __init__(self, msg: str, e: Exception | None = None) -> None:
        super().__init__(msg, e)
        self.message = msg
        self.__cause__ = e

    def __str__(self) -> str:
        return f"{self.message}Caused by "\
            f"{self.__cause__.__class__.__name__}: {self.__cause__}"


class _RegexFactory:
    _regex: 'Regex'
    _pattern: str
    _cur: int
    _last_token: State

    _capture_auto_id: int
    _cursor_started: int
    # _join_to_end: list[State]

    def __init__(self, pattern: str,
                 *, _cursor: int = 0,
                 _cid: int = 0,
                 _open_bracket_pos: int = 0) -> None:
        rx = Regex(_privated=None)
        rx.transition_table = np.empty((1, 1), dtype=set)
        self._regex = rx
        self._pattern = pattern
        self._cur = _cursor
        # Should error if accessed before assigned
        # self._last_token = 0

        self._capture_auto_id = _cid
        self._cursor_started = _open_bracket_pos
        # self._join_to_end = []

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
                raise _RegexFactory.PatternParseError(
                    f"""Could not find closing '{ch}' for opening \
'{_open_ch_map[ch]}' at position {_started_at}:
"{self._pattern}"
 {' ' * _started_at}^ here
""", e)
            else:
                raise _RegexFactory.PatternParseError(
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
        raise NotImplementedError()
        # ranges:
        # {chr(i) for i in range(ord(start), ord(end) + 1)}

    def build(self, *, _nested: bool = False) -> 'Regex':
        while self._cur < len(self._pattern):
            if self._parse_char(self._consume_char(), _nested):
                break
        return self._regex

    def _parse_escaped(self, char: str) -> None:
        match char:
            # Special chars:
            # \A, \Z, \w, \d, etc...
            case ch if ch in _parser_symbols_escaped:
                # if ch == 'A':
                #     self._anchored = True
                self._last_token = self._regex.end
                self._regex += _parser_symbols_escaped[ch]
            case (ch, _) if ch in "\\.^$+*?[]{}()":
                self._last_token = self._regex.end
                self._regex += ConsumeString(ch)
            case _:
                bk = '\\'  # dumb python
                raise PatternParseError(
                    f"Unexpected sequence: "
                    f"'{bk if self._escaped else ''}{char}'")

    def _parse_char(self, char: str, nested: bool) -> bool:
        if char == '\\':
            self._parse_escaped(self._consume_char())
            return False
        match char:
            # ^, $, ., etc...
            case ch if ch in _parser_symbols:
                self._last_token = self._regex.end
                self._regex += _parser_symbols[ch]
            case '?':  # Make _last_token optional
                self._regex.connect(self._last_token, self._regex.end,
                                    MatchConditions.epsilon_transition)
            case '+':  # Repeat 1+ times quantifier
                self._regex.connect(self._regex.end, self._last_token,
                                    MatchConditions.epsilon_transition)
                # sandbox to prevend out-of-order-ing sequenced loops
                # TODO: is this neccesary??
                # i think it is now
                self._last_token = self._regex.end
                self._regex += MatchConditions.epsilon_transition
            case '*':  # Repeat 0+ times quantifier
                # Optional
                self._regex.connect(self._last_token, self._regex.end,
                                    MatchConditions.epsilon_transition)
                # Repeating
                self._regex.connect(self._regex.end, self._last_token,
                                    MatchConditions.epsilon_transition)
                # sandbox to prevend out-of-order-ing sequenced loops
                # TODO: is this neccesary??
                # i think it is now
                self._last_token = self._regex.end
                self._regex += MatchConditions.epsilon_transition
            case '[':  # Character class specifiers
                negated = self._try_consume('^')
                cls_specifier = self._consume_till_next(
                    ']', self._is_unescaped)
                chars = SignedSet(
                    self._chars_from_char_class(cls_specifier),
                    negate=negated)
                self._last_token = self._regex.end
                self._regex += ConsumeAny(chars)
            case '{':  # n-quantifier
                quantity = self._consume_till_next('}', self._is_unescaped)
                # TODO: handle ;)
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
                inner_group = inner_builder.build(_nested=True)
                # Copy new state
                self._cur = inner_builder._cur
                self._capture_auto_id = inner_builder._capture_auto_id
                # TODO: capture groups
                # epsilon moves sandbox group and prevent loops escaping
                self._last_token = self._regex.end
                self._regex += MatchConditions.epsilon_transition
                self._regex += inner_group
                self._regex += MatchConditions.epsilon_transition
                if nested:
                    # Ensure that we still also have closing bracket
                    _ = self._find_next(')', self._is_unescaped,
                                        _started_at=self._cursor_started)
            case ')':
                if not nested:
                    # TODO: error message
                    raise PatternParseError("Unopened bracket")
                # Exit parse loop early, jump to outer group
                return True
            case '}' | ']':
                raise PatternParseError("Unopened bracket")
            case '|':  # or
                # Two options:
                # _start -> _end
                # _end -> [end of parse]
                # # Connect first option to end
                # self._join_to_end.append(self._regex.end)
                # # Start second option at _start,
                # # continue parsing from there
                # new_state = self._regex.append_state()
                # self._regex.connect(self._regex.start, new_state,
                #                     MatchConditions.epsilon_transition)
                # Parse RHS of expression
                rh_builder = _RegexFactory(
                    self._pattern,
                    _cursor=self._cur,
                    _cid=self._capture_auto_id,
                    _open_bracket_pos=self._cursor_started)
                self._cur = rh_builder._cur
                self._capture_auto_id = rh_builder._capture_auto_id
                rh_group = rh_builder.build(_nested=nested)
                self._regex |= rh_group
                return True
            # All other chars:
            case ch:
                self._last_token = self._regex.end
                self._regex += ConsumeString(ch)
        return False


class Regex:
    # S_n = x where table[S_(n-1), x].any(x=>x(ctx) is True)
    transition_table: np.ndarray[set[ParserPredicate]]

    def _expand(self, size):
        old_size = self.size
        new_size = self.size + size
        # add vertical space
        self.transition_table.resize((new_size, old_size), refcheck=False)
        temp = self.transition_table.T.copy()
        temp.resize((new_size, new_size), refcheck=False)
        self.transition_table = temp.T.copy()

    @property
    def start(self) -> State:
        return 0

    @property
    def end(self) -> State:
        assert (len(self.transition_table.shape) == 2
                and self.transition_table.shape[0]
                == self.transition_table.shape[1])
        return self.transition_table.shape[0] - 1

    @property
    def size(self) -> int:
        assert (len(self.transition_table.shape) == 2
                and self.transition_table.shape[0]
                == self.transition_table.shape[1])
        return self.transition_table.shape[0]

    @overload
    def __new__(cls, regex: str) -> Self:
        ...

    @overload
    def __new__(cls, *, _privated) -> Self:
        ...

    @overload
    def __new__(
            cls,
            transition_table: np.ndarray[set[ParserPredicate]],
            *, _privated) -> Self:
        ...

    @overload
    def __new__(
            cls,
            predicate: ParserPredicate,
            *, _privated) -> Self:
        ...

    def __new__(cls, *args, **kwargs) -> Self:
        match args, kwargs:
            case (str(regex),), {}:
                return _RegexFactory(regex).build()
            case (ParserPredicate() as x,), {"_privated": _}:
                result = super().__new__(cls)
                result.transition_table = np.empty((2, 2), dtype=set)
                result.connect(0, 1, x)
                return result
            case (np.ndarray() as x,), {"_privated": _}:
                result = super().__new__(cls)
                result.transition_table = x
                return result
            case (), {"_privated": _}:
                return super().__new__(cls)
            case _:
                raise TypeError(f"Invalid args to {cls.__name__}()", args)

    def append_state(self) -> State:
        # Resize table to have new state at end
        self._expand(1)
        return self.end

    def connect(self,
                start_state: State,
                end_state: State,
                connection: ParserPredicate) -> None:
        if not self.transition_table[start_state, end_state]:
            self.transition_table[start_state, end_state] = set()
        self.transition_table[start_state, end_state].add(connection)

    def __iadd__(self, other: Any) -> Self:
        if isinstance(other, Regex):
            other = other.copy()
            initial_end = self.end
            offset = self.size
            self._expand(other.size)
            # Connect our end to their start
            self.connect(initial_end, offset + other.start,
                         MatchConditions.epsilon_transition)
            # Copy their data into ours
            self.transition_table[offset:,
                                  offset:] = other.transition_table
        elif isinstance(other, ParserPredicate):
            old_end = self.end
            new_state = self.append_state()
            self.connect(old_end, new_state, other)
        else:
            raise NotImplementedError()
        return self

    def __ior__(self, other: 'Regex') -> Self:
        other = other.copy()
        initial_end = self.end
        offset = self.size
        self._expand(other.size)
        # Connect our start to their start
        self.connect(self.start, offset + other.start,
                     MatchConditions.epsilon_transition)
        # Connect our end to their end
        self.connect(initial_end, offset + other.end,
                     MatchConditions.epsilon_transition)
        # Copy their data into ours
        self.transition_table[offset:,
                              offset:] = other.transition_table
        return self

    def optional(self) -> Self:
        self.connect(self.start, self.end,
                     MatchConditions.epsilon_transition)
        return self

    def repeated(self) -> Self:
        self.connect(self.end, self.start,
                     MatchConditions.epsilon_transition)
        return self

    @staticmethod
    def _inner_copy_set(obj: Any):
        if isinstance(obj, set):
            return obj.copy()
        return None

    def copy(self):
        # deep copy set objs
        new_tx = np.vectorize(Regex._inner_copy_set)(self.transition_table)
        return Regex(new_tx, _privated=None)

    def __str__(self) -> str:
        return "[%s]" % ',\n '.join([
            "[%s]" % ', '.join([
                f"{{{', '.join([str(edge) for edge in edges])}}}"
                if isinstance(edges, set) else "{}"
                for edges in row])
            for row in self.transition_table])
