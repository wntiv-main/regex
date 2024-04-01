from abc import ABC, abstractmethod
from enum import IntFlag, auto
from typing import Any, Callable, Generic, Iterable, Iterator, TypeAlias, overload
from funcutil import *
from funcutil import _hash_set


class RegexFlags(IntFlag):
    GLOBAL = auto()
    MULTILINE = auto()
    CASE_SENSATIVE = auto()


_parser_symbols: dict[str, 'ParserPredicate'] = {}
_parser_symbols_escaped: dict[str, 'ParserPredicate'] = {}

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


class SignedSet(Generic[T]):
    _negate: bool
    _accept: set[T]

    def __init__(self,
                 value: 'SignedSet[T]' | Iterable[T] | None = None,
                 negate: bool = False):
        if isinstance(value, SignedSet):
            self._accept = value._accept
            self._negate = bool(negate ^ value._negate)
        else:
            self._accept = set() if value is None else set(value)
            self._negate = negate

    def __neg__(self):
        return SignedSet(self, True)

    def __contains__(self, value: T):
        return bool(self._negate ^ (value in self._accept))

    def intersection(self, other: 'SignedSet') -> 'SignedSet':
        match self._negate, other._negate:
            case False, False:
                return SignedSet(self._accept & other._accept)
            case True, True:
                return SignedSet(self._accept | other._accept, True)
            case False, True:
                return SignedSet(self._accept - other._accept)
            case True, False:
                return SignedSet(other._accept - self._accept)
    __and__ = intersection

    def symmetric_difference(self, other: 'SignedSet') -> 'SignedSet':
        match self._negate, other._negate:
            case False, False:
                return SignedSet(self._accept ^ other._accept)
            case True, True:
                return SignedSet(self._accept ^ other._accept)
            case (False, True) | (True, False):
                return SignedSet(self._accept ^ other._accept, True)
    __xor__ = symmetric_difference

    def difference(self, other: 'SignedSet') -> 'SignedSet':
        match self._negate, other._negate:
            case False, False:
                return SignedSet(self._accept - other._accept)
            case True, True:
                return SignedSet(other._accept - self._accept)
            case False, True:
                return SignedSet(self._accept & other._accept)
            case True, False:
                return SignedSet(self._accept | other._accept, True)
    __sub__ = difference

    def __hash__(self) -> int:
        return _hash_set(self._accept) ^ (self._negate * ~0)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SignedSet):
            return (self._negate == other._negate
                    and self._accept == other._accept)
        return NotImplemented

    def __bool__(self):
        return len(self._accept) > 0 or self._negate

    # not using __len__ because sometimes length is not finite
    # (represented by negative, which is not allowed by __len__)
    def length(self) -> int:
        if self._negate:
            return -len(self._accept)
        return len(self._accept)

    def unwrap_value(self) -> T:
        assert len(self) == 1
        return self._accept.copy().pop()

    def __str__(self):
        return f"{'!' if self._negate else ''}{self._accept}"


class ParserPredicate(ABC):
    @abstractmethod
    def evaluate(self, ctx: 'MatchConditions') -> bool:
        ...

    @overload
    def __call__(self, func: Callable) -> Self:
        """
        Overload to allow use as a decorator.

        Returns:
            The current instance in place of the decorated function.
        """
        ...

    @overload
    def __call__(self, ctx: 'MatchConditions') -> bool:
        ...

    def __call__(self, *args):
        match args:
            case (x,) if callable(x):
                return self
            case (MatchConditions(ctx),):
                return self.evaluate(ctx)
            case _:
                return TypeError(f"Invalid arguments to {self}()")

    @abstractmethod
    def coverage(self) -> SignedSet[str]:
        ...

    def __hash__(self) -> int:
        return hash(self.coverage())

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ParserPredicate):
            return self.coverage() == other.coverage()
        return NotImplemented


class GenericParserPredicate(ParserPredicate):
    _coverage: SignedSet[str]
    _evaluate: Callable[['MatchConditions'], bool]

    def __init__(self,
                 _coverage: SignedSet[str],
                 _evaluate: Callable[['MatchConditions'], bool]):
        self._coverage = _coverage
        self._evaluate = _evaluate

    def evaluate(self, ctx: 'MatchConditions') -> bool:
        return self._evaluate(ctx)

    def coverage(self):
        return self._coverage

    @classmethod
    def of(cls, *, coverage: SignedSet[str]) -> Callable[
            [Callable[['MatchConditions'], bool]],
            'ParserPredicate']:
        return functools.partial(cls, coverage)


class ConsumeString(ParserPredicate):
    match_string: str

    def __init__(self, match_string: str) -> None:
        self.match_string = match_string

    def evaluate(self, ctx: 'MatchConditions') -> bool:
        if ctx._string[ctx._cursor:][0:len(self.match_string)] \
                == self.match_string:
            ctx._cursor += len(self.match_string)
            return True
        return False

    def coverage(self):
        # Should never be called after concatenation
        assert len(self.match_string) == 1
        return SignedSet((self.match_string,))


class ConsumeAny(ParserPredicate):
    match_set: SignedSet[str]

    def __init__(self, match_set: SignedSet[str] | Iterable[str]) -> None:
        self.match_set = SignedSet(match_set)

    def evaluate(self, ctx: 'MatchConditions') -> bool:
        if ctx._string[ctx._cursor:][0] in self.match_set:
            ctx._cursor += 1
            return True
        return False

    def coverage(self):
        return self.match_set

    def __neg__(self) -> 'ConsumeAny':
        return ConsumeAny(-self.match_set)


class MatchConditions:
    _alpha = {chr(i) for i in range(ord('a'), ord('z') + 1)}\
        | {chr(i) for i in range(ord('A'), ord('Z') + 1)}
    _digits = {chr(i) for i in range(ord('0'), ord('9') + 1)}

    _string: str
    _cursor: int

    # def try_consume(self, *, match_string: str) -> bool:
    #     if self._string[self._cursor:][0:len(match_string)] == match_string:
    #         self._cursor += len(match_string)
    #         return True
    #     return False

    # def try_consume_any(self, *, match_set: set[str]) -> bool:
    #     if self._string[self._cursor:][0] in match_set:
    #         self._cursor += 1
    #         return True
    #     return False

    # To Python conventions,
    # How THE $*@&* am I meant to embed a link if the link is longer
    # than the line length limitation for comments, without breaking the
    # link? Here, yet another reason why line length limnitations are
    # &$#*.
    # rant over '~'
    # https://en.wikipedia.org/wiki/Nondeterministic_finite_automaton#%CE%B5-closure_of_a_state_or_set_of_states

    @GenericParserPredicate.of(coverage=SignedSet(negate=True))
    def epsilon_transition(self) -> bool:
        return True

# region regex tokens
    @represented_by(symbol="d", escaped=True)
    @ConsumeAny(_digits)
    def consume_digit(self): pass

    @represented_by("D", escaped=True)
    @-consume_digit
    def consume_not_digit(self): pass

    @represented_by("w", escaped=True)
    @ConsumeAny(_alpha | _digits | {"_"})
    def consume_alphanum(self): pass

    @represented_by("W", escaped=True)
    @-consume_alphanum
    def consume_not_alphanum(self): pass

    @represented_by("s", escaped=True)
    @ConsumeAny(" \r\n\t\v\f")
    def consume_whitespace(self): pass

    @represented_by("S", escaped=True)
    @-consume_whitespace
    def consume_not_whitespace(self): pass

    @ConsumeAny("\r\n")
    def consume_newline(self): pass

    @represented_by(".")
    @-consume_newline
    def consume_not_newline(self): pass

    @represented_by("Z", escaped=True)
    def end(self) -> bool:
        return self._cursor >= len(self._string)

    @represented_by("A", escaped=True)
    def begin(self):
        return self._cursor <= 0
# endregion regex tokens


CaptureGroup: TypeAlias = int | str

class State:
    _replaced_with: 'State | None' = None
    next: set['Edge']
    previous: set['Edge']

    def __init__(self):
        self.next = set()
        self.previous = set()

    def _connect_previous(self, edge: 'Edge'):
        if self._replaced_with is not None:
            raise RuntimeError()
        self.previous.add(edge)

    def _connect_next(self, edge: 'Edge'):
        if self._replaced_with is not None:
            raise RuntimeError()
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

    def output_diff(self, other: 'State'):
        result = self.next.copy()
        for edge in other.next:
            mod_edge = edge.clone_shallow()
            # This is unsafe make sure this edge never gets used
            mod_edge._next = edge.next
            mod_edge._previous = self
            result ^= set((mod_edge,))
        return result


class Edge(UnsafeMutable):
    _next: State = Mutable
    next: State  # type hints, impl is property()
    _previous: State = Mutable
    previous: State  # type hints, impl is property()

    _opens: set[CaptureGroup]
    _closes: set[CaptureGroup]

    _predicate: ParserPredicate = Mutable
    predicate: ParserPredicate

    @overload
    def __init__(self): ...  # epsilon transition
    @overload
    def __init__(self, predicate: ParserPredicate): ...

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

    def predicate_intersection(self, other: 'Edge')\
            -> None | tuple[ParserPredicate | None,
                            ParserPredicate,
                            ParserPredicate | None]:
        self_coverage = self.predicate.coverage()
        other_coverage = other.predicate.coverage()
        if not (self_coverage & other_coverage):
            return None
        string_matches = (ConsumeAny,
                          ConsumeString,
                          GenericParserPredicate)
        if (isinstance(self.predicate, string_matches)
                and isinstance(other.predicate, string_matches)):
            # Left side
            left_set = self_coverage - other_coverage
            if not left_set:
                left = None
            elif left_set.length() == 1:
                left = ConsumeString(left_set.unwrap_value())
            else:
                left = ConsumeAny(left_set)
            # Right side
            right_set = other_coverage - self_coverage
            if not right_set:
                right = None
            elif right_set.length() == 1:
                right = ConsumeString(right_set.unwrap_value())
            else:
                right = ConsumeAny(right_set)
            # Intersect
            intersect_set = self_coverage & other_coverage
            if intersect_set.length() == 1:
                intersect = ConsumeString(intersect_set.unwrap_value())
            else:
                intersect = ConsumeAny(intersect_set)
            return left, intersect, right
        raise NotImplementedError()

    # Deduce str representation of predicate function
    def _predicate_str(self):
        # Trivial cases
        for k, v in _parser_symbols_escaped.items():
            if v == self.predicate:
                return f"\\{k}"
        for k, v in _parser_symbols.items():
            if v == self.predicate:
                return k
        # less sketchy predicate parsing
        if (self.predicate == MatchConditions.epsilon_transition):
            return "\u03B5"  # Epsilon char
        if isinstance(self.predicate, ConsumeString):
            return f"'{self.predicate.match_string}'"
        if isinstance(self.predicate, ConsumeAny):
            return str(self.predicate.match_set)
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

    def remove(self, *, chain: bool = False):
        if chain and self.next.inputs() < 1:
            while len(self.next.next):
                with self.next.next.pop() as edge:
                    edge.remove(chain=True)
        self.next = self.previous = None

    def is_free(self) -> bool:
        return (self.predicate == MatchConditions.epsilon_transition
                and not (self._opens or self._closes))

    def clone_shallow(self) -> 'Edge':
        new = Edge(self.predicate)
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

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Edge): return NotImplemented
        return id(self) == id(other) or (
            hash(self) == hash(other)
            and self.next == other.next
            and self.previous == other.previous
            and self._opens == other._opens
            and self._closes == other._closes
            and self.predicate == other.predicate)

    def __hash__(self) -> int:
        return hash((
            self.next,
            self.previous,
            _hash_set(self._opens),
            _hash_set(self._closes),
            self.predicate))
