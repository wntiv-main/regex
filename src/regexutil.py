from enum import IntFlag, auto
from typing import Callable, Iterator, TypeAlias, overload
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

    def outputs(self) -> int:
        return len(self.next)

    def inputs(self) -> int:
        return len(self.previous)

    def merge(self, other: 'State'):
        for edge in other.previous.copy():
            with edge:
                if edge.previous == self and edge.is_free():
                    edge.remove()
                else:
                    edge.next = self
        for edge in other.next.copy():
            with edge:
                edge.previous = self
        other._replaced_with = self

    def clone_shallow(self, *, reverse: bool = True) -> 'State':
        new = State()
        for edge in self.next.copy():
            with edge.clone_shallow() as new_edge:
                # if edge.next == self:
                #     new_edge.next = new
                # else:
                new_edge.next = edge.next
                new_edge.previous = new
        if reverse:
            for edge in self.previous.copy():
                with edge.clone_shallow() as new_edge:
                    if edge.previous == self:
                        new_edge.previous = new
                    else:
                        new_edge.previous = edge.previous
                    new_edge.next = new
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


class Edge(UnsafeMutable):
    _next: State = Mutable()
    next: State  # type hints, impl is property()
    _previous: State = Mutable()
    previous: State  # type hints, impl is property()

    _opens: set[CaptureGroup]
    _closes: set[CaptureGroup]

    _predicate: MatchPredicate

    @overload
    def __init__(self): ...  # epsilon transition
    @overload
    def __init__(self, predicate: MatchPredicate): ...

    def __init__(self, *args) -> None:
        super().__init__()
        self._next = None
        self._previous = None
        self._opens = set()
        self._closes = set()
        match args:
            case ():
                self._predicate = MatchConditions.epsilon_transition
            case (predicate,) if callable(predicate):
                self._predicate = predicate
            case _:
                raise TypeError(
                    f"Inproper args to new {self.__class__.__name__}()")

    def __enter__(self) -> 'Edge':
        if self._next is not None:
            self._next._disconnect_previous(self)
        if self._previous is not None:
            self._previous._disconnect_next(self)
        return super().__enter__()

    def __exit__(self, *args) -> None:
        if self._next is not None:
            self._next._connect_previous(self)
        if self._previous is not None:
            self._previous._connect_next(self)
        return super().__exit__(*args)

    # Deduce str representation of predicate function
    def _predicate_str(self):
        # Trivial cases
        for k, v in _parser_symbols_escaped.items():
            if v == self._predicate:
                return f"\\{k}"
        for k, v in _parser_symbols.items():
            if v == self._predicate:
                return k
        # sketchy predicate parsing
        if (hasattr(self._predicate, '__name__')
                and self._predicate.__name__
                == MatchConditions.epsilon_transition.__name__):
            return "\u03B5"  # Epsilon char
        if isinstance(self._predicate, functools.partial):
            func = self._predicate.func
            if (hasattr(func, '__name__')
                    and func.__name__ == MatchConditions.try_consume.__name__):
                return f"'{self._predicate.keywords['match_char']}'"
        # TODO: handle more cases
        return "Unknown"

    # Debug representation for DebugGraphViewer
    def __repr__(self) -> str:
        result = f"{self._predicate_str()}"
        if self._opens or self._closes:
            result += f", ({self._opens};{self._closes})"
        return result

    @UnsafeMutable.mutator
    def open(self, group_id: CaptureGroup):
        self._opens.add(group_id)

    @UnsafeMutable.mutator
    def move_opens(self) -> Iterator[CaptureGroup]:
        while len(self._opens):
            yield self._opens.pop()

    def has_opens(self):
        return bool(len(self._opens))

    def has_closes(self):
        return bool(len(self._closes))

    @UnsafeMutable.mutator
    def close(self, group_id: CaptureGroup):
        self._closes.add(group_id)

    @UnsafeMutable.mutator
    def move_closes(self) -> Iterator[CaptureGroup]:
        while len(self._closes):
            yield self._closes.pop()

    def remove(self):
        self.next = self.previous = None

    def is_free(self) -> bool:
        return (self._predicate == MatchConditions.epsilon_transition
                and not (self._opens or self._closes))

    def clone_shallow(self) -> 'Edge':
        new = Edge()
        new._predicate = self._predicate
        new._closes = self._closes
        new._opens = self._opens
        return new

    def clone(
            self,
            map_state: dict['Edge', 'Edge'],
            map_path: dict['Edge', 'Edge']) -> 'Edge':
        if (self in map_path):
            return map_path[self]
        with Edge() as new:
            new.next = self.next.clone(map_state, map_path)
            new.previous = self.previous.clone(map_state, map_path)
            new.opens = self.opens
            new.closes = self.closes
            new.predicate = self.predicate
        map_path[self] = new
        return new

    def predicates_match(self, other: 'Edge') -> bool:
        if self._predicate == other._predicate:
            return True
        if (isinstance(self._predicate, functools.partial)
                and isinstance(other._predicate, functools.partial)
                and self._predicate.func == other._predicate.func
                and self._predicate.args == other._predicate.args
                and self._predicate.keywords == other._predicate.keywords):
            return True
        return False

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Edge): return NotImplemented
        return id(self) == id(other) or (
                hash(self) == hash(other)
                and self.next == other.next
                and self.previous == other.previous
            and self._opens == other._opens
            and self._closes == other._closes
                and self.predicates_match(other))

    @staticmethod
    def _hash_set(value: set) -> int:
        result = 0
        for element in value:
            result ^= hash(element)
        return result

    def __hash__(self) -> int:
        return hash((
            self.next,
            self.previous,
            Edge._hash_set(self._opens),
            Edge._hash_set(self._closes),
            self._predicate))
