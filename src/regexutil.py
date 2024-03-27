from enum import IntFlag, auto
from typing import Callable, TypeAlias, overload
from funcutil import *


class RegexFlags(IntFlag):
    GLOBAL = auto()
    MULTILINE = auto()
    CASE_SENSATIVE = auto()


_parser_symbols: dict[str, Callable[['MatchConditions'], bool]] = {}
_parser_symbols_escaped: dict[str, Callable[['MatchConditions'], bool]] = {}

T = TypeVar("T")
TArgs = TypeVarTuple("TArgs")


class represented_by:
    _symbol: str
    _escaped: bool

    def __init__(self, symbol: str, *, escaped: bool = False):
        self._symbol = symbol
        self._escaped = escaped

    def __call__(self, func: Callable[[*TArgs], T]) -> Callable[[*TArgs], T]:
        if self._escaped:
            _parser_symbols_escaped[self._symbol] = func
        else:
            _parser_symbols[self._symbol] = func
        return func


class MatchConditions:
    _alpha = {chr(i) for i in range(ord('a'), ord('z') + 1)}\
        | {chr(i) for i in range(ord('A'), ord('Z') + 1)}
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

    # To Python conventions,
    # How THE $*@&* am I meant to embed a link if the link is longer
    # than the line length limitation for comments, without breaking the
    # link? Here, yet another reason why line length limnitations are
    # &$#*.
    # rant over '~'
    # https://en.wikipedia.org/wiki/Nondeterministic_finite_automaton#%CE%B5-closure_of_a_state_or_set_of_states
    def epsilon_transition(self) -> bool:
        return True

# region regex tokens
    @represented_by(symbol="d", escaped=True)
    @extend(try_consume_any, _digits)
    def consume_digit(self): pass

    @represented_by("D", escaped=True)
    @negate
    @extend(consume_digit)
    def consume_not_digit(self): pass

    @represented_by("w", escaped=True)
    @extend(try_consume_any, _alpha | _digits | {"_"})
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


MatchPredicate: TypeAlias = Callable[[MatchConditions], bool]

CaptureGroup: TypeAlias = int | str

class State:
    _replaced_with: 'State | None' = None
    next: set['Edge']
    previous: set['Edge']

    def __init__(self):
        self.next = set()
        self.previous = set()

    def _connect_previous(self, edge: 'Edge'):
        self.previous.add(edge)

    def _connect_next(self, edge: 'Edge'):
        self.next.add(edge)

    def _disconnect_previous(self, edge: 'Edge'):
        self.previous.discard(edge)

    def _disconnect_next(self, edge: 'Edge'):
        self.next.discard(edge)

    def merge(self, other: 'State'):
        for edge in other.previous.copy():
            if edge.previous == self and edge.is_free():
                edge.remove()
                continue
            new_edge = edge.clone_shallow()
            new_edge.rconnect(edge.previous)
            new_edge.connect(self)
            edge.remove()
        for edge in other.next.copy():
            new_edge = edge.clone_shallow()
            new_edge.connect(edge.next)
            new_edge.rconnect(self)
            edge.remove()
        other._replaced_with = self

    def clone_shallow(self, *, reverse: bool = True) -> 'State':
        new = State()
        for edge in self.next.copy():
            new_edge = edge.clone_shallow()
            if edge.next == self:
                new_edge.connect(new)
            else:
                new_edge.connect(edge.next)
            new_edge.rconnect(new)
        if reverse:
            for edge in self.previous.copy():
                new_edge = edge.clone_shallow()
                if edge.previous == self:
                    new_edge.rconnect(new)
                else:
                    new_edge.rconnect(edge.previous)
                new_edge.connect(new)
        return new

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

    opens: set[CaptureGroup]
    closes: set[CaptureGroup]

    predicate: MatchPredicate

    @overload
    def __init__(self): ...  # epsilon transition
    @overload
    def __init__(self, predicate: MatchPredicate): ...

    def __init__(self, *args) -> None:
        self.next = None
        self.previous = None
        self.opens = set()
        self.closes = set()
        match args:
            case ():
                self.predicate = MatchConditions.epsilon_transition
            case (predicate,) if callable(predicate):
                self.predicate = predicate
            case _:
                raise TypeError(
                    f"Inproper args to new {self.__class__.__name__}()")

    # Deduce str representation of predicate function
    def _predicate_str(self):
        # Trivial cases
        for k, v in _parser_symbols_escaped.items():
            if v == self.predicate:
                return f"\\{k}"
        for k, v in _parser_symbols.items():
            if v == self.predicate:
                return k
        # sketchy predicate parsing
        if (hasattr(self.predicate, '__name__')
                and self.predicate.__name__
                == MatchConditions.epsilon_transition.__name__):
            return "\u03B5"  # Epsilon char
        if isinstance(self.predicate, functools.partial):
            func = self.predicate.func
            if (hasattr(func, '__name__')
                    and func.__name__ == MatchConditions.try_consume.__name__):
                return f"'{self.predicate.keywords['match_char']}'"
        # TODO: handle more cases
        return "Unknown"

    # Debug representation for DebugGraphViewer
    def __repr__(self) -> str:
        result = f"{self._predicate_str()}"
        if self.opens or self.closes:
            result += f", ({self.opens};{self.closes})"
        return result

    def unsafe_connect(self, state: State):
        self.next = state
        if self.previous is not None:
            state._connect_previous(self)
            self.previous._connect_next(self)

    def connect(self, state: State):
        if self.next is not None:
            raise RuntimeError("handle better")
        self.unsafe_connect(state)

    def unsafe_rconnect(self, state: State):
        self.previous = state
        if self.next is not None:
            state._connect_next(self)
            self.next._connect_previous(self)

    def rconnect(self, state: State):
        if self.previous is not None:
            raise RuntimeError("handle better")
        self.unsafe_rconnect(state)

    def remove(self):
        self.previous._disconnect_next(self)
        self.next._disconnect_previous(self)
        self.next = self.previous = None

    def is_free(self) -> bool:
        return (self.predicate == MatchConditions.epsilon_transition
                and not (self.opens or self.closes))

    def clone_shallow(self) -> 'Edge':
        new = Edge()
        new.predicate = self.predicate
        new.closes = self.closes
        new.opens = self.opens
        return new

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

    def predicates_match(self, other: 'Edge') -> bool:
        if self.predicate == other.predicate:
            return True
        if (isinstance(self.predicate, functools.partial)
                and isinstance(other.predicate, functools.partial)
                and self.predicate.func == other.predicate.func
                and self.predicate.args == other.predicate.args
                and self.predicate.keywords == other.predicate.keywords):
            return True
        return False

    def __equals__(self, other: 'Edge') -> bool:
        return (self.value_hash() == other.value_hash()
                and self.next == other.next
                and self.previous == other.previous
                and self.opens == other.opens
                and self.closes == other.closes
                and self.predicates_match(other))

    @staticmethod
    def _hash_set(value: set) -> int:
        result = 0
        for element in value:
            result ^= hash(element)
        return result

    def __hash__(self) -> int:
        return (hash(self.next)
                ^ (hash(self.previous) >> 1)
                ^ Edge._hash_set(self.opens)
                ^ (Edge._hash_set(self.closes) >> 1)
                ^ hash(self.predicate))
