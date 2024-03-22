from enum import IntFlag, auto
import functools
from typing import Callable, overload

from funcutil import *

if __name__ == "__main__":
    print("WARNING: This package is made to be imported not run directly")
    try:
        import networkx
        import matplotlib.pyplot as plt
    except ImportError:
        print("If you wish to proceed to display the debug graphic, "
              "`$ pip install networkx` and `$ pip install matplotlib`")


class RegexFlags(IntFlag):
    GLOBAL = auto()
    MULTILINE = auto()
    CASE_SENSATIVE = auto()


parser_symbols: dict[str, Callable[['MatchStream'], bool]] = {}
parser_symbols_escaped: dict[str, Callable[['MatchStream'], bool]] = {}


def represented_by(
        func: Callable[['MatchStream'], bool],
        symbol: str,
        *, escaped: bool = False):
    if escaped:
        parser_symbols_escaped[symbol] = func
    else:
        parser_symbols[symbol] = func
    return func


class MatchStream:
    _alpha = {chr(i) for i in range(ord('a'), ord('z') + 1)}
    _digits = {chr(i) for i in range(ord('0'), ord('9') + 1)}

    _string: str
    _cursor: int

    def try_consume(self, *, match_string: str) -> bool:
        if self._string[self._cursor:][0:len(match_string)] == match_string:
            self._cursor += len(match_string)
            return True
        return False
    
    def try_consume_any(self, *, match_set: set[str]) -> bool:
        if self._string[self._cursor:][0] in match_set:
            self._cursor += 1
            return True
        return False
    
    # https://en.wikipedia.org/wiki/Nondeterministic_finite_automaton#%CE%B5-closure_of_a_state_or_set_of_states
    def epsilon_transition(self) -> bool:
        return True

# region regex tokens
    @represented_by("d", escaped=True)
    @extend(try_consume_any, _digits)
    def consume_digit(self): pass

    @represented_by("D", escaped=True)
    @negate
    @extend(consume_digit)
    def consume_not_digit(self): pass

    @represented_by("w", escaped=True)
    @extend(try_consume_any, _alpha + _digits + {"_"})
    def consume_alphanum(self): pass

    @represented_by("W", escaped=True)
    @negate
    @extend(consume_alphanum)
    def consume_not_alphanum(self): pass

    @represented_by("s", escaped=True)
    @extend(try_consume_any, set(" \r\n\t\v\f"))
    def consume_whitespace(self): pass

    @represented_by("S", escaped=True)
    @negate
    @extend(consume_whitespace)
    def consume_not_whitespace(self): pass

    @extend(try_consume_any, set("\r\n"))
    def consume_newline(self): pass

    @represented_by(".")
    @negate
    @extend(consume_newline)
    def consume_not_newline(self): pass

    @represented_by("Z", escaped=True)
    @represented_by("$")
    def end(self) -> bool:
        return self._cursor >= len(self._string)

    @represented_by("A", escaped=True)
    @represented_by("^")
    def begin(self):
        return self._cursor <= 0
# endregion regex tokens


class CaptureGroup:
    _groups = {}
    id: int | str

    @staticmethod
    def group_for(id: int | str):
        if id in CaptureGroup._groups:
            return CaptureGroup._groups[id]
        else:
            new_group = CaptureGroup(id)
            CaptureGroup._groups[id] = new_group 
            return new_group

class State:
    next: set['Edge']
    previous: set['Edge']

    def __init__(self):
        self.next = {}
        self.previous = {}

    def clone(
            self,
            map_state: dict['State', 'State'],
            map_path: dict['Edge', 'Edge']) -> 'State':
        if (self in map_state):
            return map_state[self]
        new = State()
        for path in self.next:
            new.next.add(path.clone(map_state, map_path))
        for path in self.previous:
            new.previous.add(path.clone(map_state, map_path))
        map_state[self] = new
        return new


class Edge:
    next: State
    previous: State

    opens: list[CaptureGroup]
    closes: list[CaptureGroup]

    predicate: Callable[[MatchStream], bool]

    @overload
    def __init__(self): ...  # epsilon transition
    @overload
    def __init__(self, predicate: Callable[[MatchStream], bool]): ...

    def __init__(self, *args) -> None:
        match args:
            case ():
                self.predicate = MatchStream.epsilon_transition
            case (Callable(predicate),):
                self.predicate = predicate
            case _:
                raise TypeError("Inproper args to new Path()")

    def clone(
            self,
            map_state: dict['Edge', 'Edge'],
            map_path: dict['Edge', 'Edge']) -> 'Edge':
        if (self in map_path):
            return map_path[self]
        new = Edge()
        new.next = self.next.clone(map_state, map_path)
        new.previous = self.previous.clone(map_state, map_path)
        new.opens = self.opens
        new.closes = self.closes
        new.predicate = self.predicate
        map_path[self] = new
        return new


class RegexBuilder:
    class PatternParseError(Exception):
        pass

    _special_chars = "\\.^$+*?[]{}()"

    _pattern: str
    _cursor: int

    _escaped: bool
    _start: State
    _end: State
    _path_ends: list[Edge]

    def __init__(self, pattern: str):
        self._pattern = pattern
        self._cursor = 0

        self._escaped = False
        self._path_ends = []
        self._start = State()

    def _append_path(self, path: Edge):
        for end in self._path_ends:
            end.next.append(path)
            path.previous.append(path)

    def _consume_char(self) -> str:
        ch = self._pattern[self._cursor]
        self._cursor += 1
        return ch

    def _try_consume(self, match: str) -> bool:
        if self._pattern[self._cursor:].startswith(match):
            self._cursor += len(match)
            return True
        return False

    def build(self) -> Edge:
        while self._cursor < len(self._pattern):
            self._parse_char(self._consume_char())

    def _parse_char(self, char):
        match char, self._escaped:
            case '\\', False:
                self._escaped = True
                return  # dont reset `escaped` flag
            # \A, \Z, \w, \d, etc...
            case ch, True if ch in parser_symbols_escaped:
                self._append_path(Edge(parser_symbols_escaped[ch]))
            # ^, $, ., etc...
            case ch, False if ch in parser_symbols:
                self._append_path(Edge(parser_symbols[ch]))
            # all other chars
            case (ch, _) if not self._escaped or ch in self._special_chars:
                self._append_path(Edge(functools.partial(
                    MatchStream.try_consume, match_char=ch)))
            case _:
                raise RegexBuilder.PatternParseError()
        self._escaped = False


def main():
    RegexBuilder(r"\w+@\w+\.").build()


if __name__ == "__name__":
    main()
