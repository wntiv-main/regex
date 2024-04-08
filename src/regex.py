__all__ = ["PatternParseError", "Regex", "State"]
__author__ = "Callum Hynes"

from typing import Any, Callable, Self, Sequence, TypeAlias, overload
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
    _last_token: 'Regex | None'

    _capture_auto_id: int
    _cursor_started: int

    def __init__(self, pattern: str,
                 *, _cursor: int = 0,
                 _cid: int = 0,
                 _open_bracket_pos: int = 0) -> None:
        rx = Regex(_privated=None)
        rx.transition_table = Regex._empty_arr((1, 1))
        self._regex = rx
        self._pattern = pattern
        self._cur = _cursor
        # Should error if accessed before assigned
        self._last_token = None

        self._capture_auto_id = _cid
        self._cursor_started = _open_bracket_pos

    def _append(self, connection: 'ParserPredicate | Regex') -> None:
        if isinstance(connection, ParserPredicate):
            connection = Regex(connection, _privated=None)
        if self._last_token is not None:
            self._regex += self._last_token
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
        self._regex += self._last_token
        # if not _nested:
        #     self._regex._epsilon_closure()
        return self._regex

    def _parse_escaped(self, char: str) -> None:
        match char:
            # Special chars:
            # \A, \Z, \w, \d, etc...
            case ch if ch in _parser_symbols_escaped:
                # if ch == 'A':
                #     self._anchored = True
                self._append(_parser_symbols_escaped[ch])
            case (ch, _) if ch in "\\.^$+*?[]{}()":
                self._append(ConsumeString(ch))
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
                self._append(_parser_symbols[ch])
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
                inner_group = inner_builder.build(_nested=True)
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
                if not nested:
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
                rh_group = rh_builder.build(_nested=nested)
                self._regex |= rh_group
                return True
            # All other chars:
            case ch:
                self._append(ConsumeString(ch))
        return False


class Regex:
    # S_n = x where table[S_(n-1), x].any(x=>x(ctx) is True)
    transition_table: np.ndarray[set[ParserPredicate]]
    start: State
    end: State

    @overload
    def __new__(cls, regex: str) -> Self:
        ...

    @overload
    def __new__(cls, *, _privated: None) -> Self:
        ...

    @overload
    def __new__(
            cls,
            other: 'Regex') -> Self:
        ...

    @overload
    def __new__(
            cls,
            predicate: ParserPredicate,
            *, _privated: None) -> Self:
        ...

    def __new__(cls, *args, **kwargs) -> Self:
        match args, kwargs:
            case (str(regex),), {}:
                return _RegexFactory(regex).build()
            case (ParserPredicate() as x,), {"_privated": _}:
                result = super().__new__(cls)
                result.transition_table = Regex._empty_arr((2, 2))
                result.start = 0
                result.end = 1
                result.connect(result.start, result.end, x)
                return result
            case (Regex() as x,), {}:
                result = super().__new__(cls)
                # Deep copy sets within table
                result.transition_table = np.vectorize(
                    Regex._inner_copy_set)(x.transition_table)
                result.start = x.start
                result.end = x.end
                return result
            case (), {"_privated": _}:
                result = super().__new__(cls)
                result.transition_table = Regex._empty_arr((1, 1))
                result.start = 0
                result.end = 0
                return result
            case _:
                raise TypeError(f"Invalid args to {cls.__name__}()", args)

    @property
    def size(self) -> int:
        assert (len(self.transition_table.shape) == 2
                and self.transition_table.shape[0]
                == self.transition_table.shape[1])
        return self.transition_table.shape[0]
    __len__ = size.fget

    @staticmethod
    def _empty_arr(size: Sequence[int]):
        # if only np.fromfunction() worked :(
        return np.vectorize(lambda _: set())(np.empty(size, dtype=set))

    def append_state(self) -> State:
        # Resize table to have new state at end
        self._diagonal_block_with(Regex._empty_arr((1, 1)))
        return self.size - 1

    def connect(self,
                start_state: State,
                end_state: State,
                connection: ParserPredicate) -> None:
        if not self.transition_table[start_state, end_state]:
            self.transition_table[start_state, end_state] = set()
        self.transition_table[start_state, end_state].add(connection)

    def connect_many(self,
                     start_state: State,
                     end_state: State,
                     connections: set[ParserPredicate]) -> None:
        if not self.transition_table[start_state, end_state]:
            self.transition_table[start_state, end_state] = set()
        self.transition_table[start_state, end_state] |= connections

    def _epsilon_closure(self):
        for i in range(self.size):
            for j in range(self.size):
                # TODO: soon edges will have more info
                if (MatchConditions.epsilon_transition
                        in self.transition_table[i, j]):
                    self._merge_outputs(i, j)
                    self.transition_table[i, j].remove(
                        MatchConditions.epsilon_transition)

    def _minimisation(self):
        to_remove: set[State] = set()
        for i in range(self.size - 1):
            for j in range(i + 1, self.size):
                # TODO: more robust comparison
                if self.transition_table[i] == self.transition_table[j]:
                    self._merge(i, j)
                    to_remove.add(j)
        # Remove in reverse to avoid deletions mis-ordering the matrix
        for state in sorted(to_remove, reverse=True):
            self._remove_state(state)

    def _remove_state(self, state: State) -> None:
        # remove both row and column for `state`
        self.transition_table = np.delete(
            np.delete(
                self.transition_table,
                state, 0),
            state, 1)

    def _merge_outputs(self, s1_idx: State, s2_idx: State) -> None:
        # Iterate s2 row and make same connections from s1
        it = np.nditer(self.transition_table[s2_idx, :],
                       flags=['c_index', 'refs_ok'])
        for edges in it:
            self.connect_many(s1_idx, it.index, edges)

    def _merge_inputs(self, s1_idx: State, s2_idx: State) -> None:
        # Iterate s2 column and make same connections to s1
        it = np.nditer(self.transition_table[:, s2_idx],
                       flags=['c_index', 'refs_ok'])
        for edges in it:
            self.connect_many(it.index, s1_idx, edges)

    def _merge(self, s1_idx: State, s2_idx: State) -> None:
        self._merge_inputs(s1_idx, s2_idx)
        self._merge_outputs(s1_idx, s2_idx)

    def _diagonal_block_with(self, other: np.ndarray):
        # constructs block matrix like:
        # [[self,  empty]
        #  [empty, other]]
        self.transition_table = np.block([
            [self.transition_table, Regex._empty_arr((self.size,
                                                      other.shape[1]))],
            [Regex._empty_arr((other.shape[0], self.size)), other]])

    def __iadd__(self, other: Any) -> Self:
        if isinstance(other, Regex):
            other = other.copy()
            offset = self.size
            self._diagonal_block_with(other.transition_table)
            # Connect our end to their start
            self.connect(self.end, offset + other.start,
                         MatchConditions.epsilon_transition)
            self.end = offset + other.end
        elif isinstance(other, ParserPredicate):
            new_state = self.append_state()
            self.connect(self.end, new_state, other)
            self.end = new_state
        else:
            raise NotImplementedError()
        return self

    def __ior__(self, other: 'Regex') -> Self:
        other = other.copy()
        offset = self.size
        self._diagonal_block_with(other.transition_table)
        # Connect our start to their start
        self.connect(self.start, offset + other.start,
                     MatchConditions.epsilon_transition)
        # Connect our end to their end
        self.connect(self.end, offset + other.end,
                     MatchConditions.epsilon_transition)
        self.end = offset + other.end
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
        return Regex(self)

    def __str__(self) -> str:
        return "[%s]: %d -> %d" % (',\n '.join([
            "[%s]" % ', '.join([
                f"{{{', '.join([str(edge) for edge in edges])}}}"
                if isinstance(edges, set) else "{}"
                for edges in row])
            for row in self.transition_table]), self.start, self.end)
