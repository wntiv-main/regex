from abc import ABC, abstractmethod
import functools
from typing import Any, Callable, Generic, Iterable, Optional, Self, TypeAlias, TypeVar, TypeVarTuple, overload
from funcutil import _hash_set


_parser_symbols: dict[str, 'ParserPredicate'] = {}
_parser_symbols_escaped: dict[str, 'ParserPredicate'] = {}

T = TypeVar("T")
TArgs = TypeVarTuple("TArgs")


State: TypeAlias = int


class represented_by:
    _symbol: str
    _escaped: bool

    def __init__(self, symbol: str, *, escaped: bool = False):
        self._symbol = symbol
        self._escaped = escaped

    def __call__(self, func: 'ParserPredicate') -> 'ParserPredicate':
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

    def add(self, el: T) -> None:
        if self._negate:
            self._accept.discard(el)
        else:
            self._accept.add(el)

    def discard(self, el: T) -> None:
        if self._negate:
            self._accept.add(el)
        else:
            self._accept.discard(el)

    def negate(self) -> Self:
        self._negate = not self._negate
        return self

    def __neg__(self) -> 'SignedSet':
        return self.copy().negate()

    def __contains__(self, value: T):
        return (self._negate ^ (value in self._accept))

    @overload
    @staticmethod
    def union(*sets: 'SignedSet') -> 'SignedSet':
        ...

    @overload
    def union(self, *others: 'SignedSet') -> 'SignedSet':
        ...

    def union(*sets):
        return SignedSet().i_union(*sets)
    __or__ = union

    def i_union(self, *sets: 'SignedSet') -> Self:
        for el in sets:
            match self._negate, el._negate:
                case False, False:
                    self._accept |= el._accept
                case True, True:
                    self._accept &= el._accept
                case False, True:
                    self._negate = True
                    self._accept = el._accept - self._accept
                case True, False:
                    self._accept -= el._accept
        return self
    __ior__ = i_union

    # TODO: housekeeeping: make the following more like union??
    @overload
    @staticmethod
    def intersection(*sets: 'SignedSet') -> 'SignedSet':
        ...

    @overload
    def intersection(self, *others: 'SignedSet') -> 'SignedSet':
        ...

    def intersection(*sets):
        return SignedSet(negate=True).i_intersection(*sets)
    __and__ = intersection

    def i_intersection(self, *sets: 'SignedSet') -> Self:
        for el in sets:
            match self._negate, el._negate:
                case False, False:
                    self._accept &= el._accept
                case True, True:
                    self._accept |= el._accept
                case False, True:
                    self._accept -= el._accept
                case True, False:
                    self._negate = False
                    self._accept = el._accept - self._accept
        return self
    __iand__ = i_intersection

    def symmetric_difference(self, other: 'SignedSet') -> 'SignedSet':
        match self._negate, other._negate:
            case (False, False) | (True, True):
                return SignedSet(self._accept ^ other._accept)
            case (False, True) | (True, False):
                return SignedSet(self._accept ^ other._accept, True)
    __xor__ = symmetric_difference

    def i_symmetric_difference(self, other: 'SignedSet') -> Self:
        match self._negate, other._negate:
            case False, False:
                self._accept ^= other._accept
                return SignedSet(self._accept ^ other._accept)
            case True, True:
                self._negate = False
                self._accept ^= other._accept
            case (False, True) | (True, False):
                self._negate = True
                self._accept ^= other._accept
        return self
    __ixor__ = i_symmetric_difference

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

    def i_difference(self, other: 'SignedSet') -> Self:
        match self._negate, other._negate:
            case False, False:
                self._accept -= other._accept
            case True, True:
                self._negate = False
                self._accept = other._accept - self._accept
            case False, True:
                self._accept &= other._accept
            case True, False:
                self._accept |= other._accept
        return self
    __isub__ = i_difference

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
        assert self.length() == 1
        return self._accept.copy().pop()

    def copy(self) -> 'SignedSet':
        return SignedSet(self._accept.copy(), self._negate)

    def __str__(self):
        return f"{'-' if self._negate else ''}{self._accept}"

    def __repr__(self):
        return (f"{self.__class__.__name__}"
                f"({self._accept!r}, {self._negate!r})")


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
                return self.copy()
            case (MatchConditions() as ctx,):
                return self.evaluate(ctx)
            case _:
                return TypeError(f"Invalid arguments to {self}()")

    @abstractmethod
    def coverage(self) -> SignedSet[str]:
        raise NotImplementedError()

    @abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def mutable_hash(self) -> int:
        """
        An alternative hash method that may change if the object is
        mutated.

        Returns:
            The hash of the object.
        """
        raise NotImplementedError()

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        raise NotImplementedError()

    def _symbol(self) -> Optional[str]:
        # Trivial cases
        for k, v in _parser_symbols_escaped.items():
            if v == self:
                return f"\\{k}"
        for k, v in _parser_symbols.items():
            if v == self:
                return k
        return None

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def copy(self) -> 'ParserPredicate':
        raise NotImplementedError()


class GenericParserPredicate(ParserPredicate):
    _sym: str
    _coverage: SignedSet[str]
    _evaluate: Callable[['MatchConditions'], bool]

    def __init__(self,
                 _symbol: str,
                 _coverage: SignedSet[str],
                 _evaluate: Callable[['MatchConditions'], bool]):
        self._sym = _symbol
        self._coverage = _coverage
        self._evaluate = _evaluate

    def evaluate(self, ctx: 'MatchConditions') -> bool:
        return self._evaluate(ctx)

    def coverage(self):
        return self._coverage

    @classmethod
    def of(cls, *, symbol: str, coverage: SignedSet[str]) -> Callable[
            [Callable[['MatchConditions'], bool]],
            'ParserPredicate']:
        return functools.partial(cls, symbol, coverage)

    def __hash__(self) -> int:
        return id(self._evaluate)
    mutable_hash = __hash__

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, GenericParserPredicate):
            return self._evaluate == other._evaluate
        return NotImplemented

    def __str__(self) -> str:
        return self._symbol() or self._sym

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"_symbol={self._sym!r}, "
                f"_coverage={self._coverage!r}, "
                f"_evaluate={self._evaluate!r})")
        
    def copy(self) -> 'GenericParserPredicate':
        return GenericParserPredicate(self._sym, self._coverage, self._evaluate)


class ConsumeString(ParserPredicate):
    match_string: str

    def __init__(self, match_string: str) -> None:
        self.match_string = match_string

    def evaluate(self, ctx: 'MatchConditions') -> bool:
        if (ctx._cursor + len(self.match_string) <= len(ctx._string)
            and ctx._string[ctx._cursor:][0:len(self.match_string)]
                == self.match_string):
            ctx._cursor += len(self.match_string)
            return True
        return False

    def coverage(self):
        # Should never be called after concatenation
        assert len(self.match_string) == 1
        return SignedSet((self.match_string,))

    def __hash__(self) -> int:
        return hash(self.match_string)
    mutable_hash = __hash__

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ConsumeString):
            return self.match_string == other.match_string
        if isinstance(other, ConsumeAny):
            return (other.match_set.length() == 1
                    and other.match_set.unwrap_value() == self.match_string)
        return NotImplemented

    def __str__(self) -> str:
        return self._symbol() or f"'{self.match_string}'"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.match_string!r})"
    
    def copy(self) -> 'ConsumeString':
        return ConsumeString(self.match_string)


class ConsumeAny(ParserPredicate):
    match_set: SignedSet[str]

    def __init__(self, match_set: SignedSet[str] | Iterable[str]) -> None:
        self.match_set = SignedSet(match_set)

    def evaluate(self, ctx: 'MatchConditions') -> bool:
        if (not ctx.end(ctx) and
                ctx._string[ctx._cursor] in self.match_set):
            ctx._cursor += 1
            return True
        return False

    def coverage(self):
        return self.match_set

    def __neg__(self) -> 'ConsumeAny':
        return ConsumeAny(-self.match_set)

    def __hash__(self) -> int:
        # Lets not use hash in case of mutation :)
        # Note this means that a == b does NOT imply hash(a) == hash(b)
        # as is common expectation. lets hope noone notices :/
        # Otherwise, we *could* implement a secondary .val_equals()
        # method, but im lazy :P
        return id(self.match_set)

    def mutable_hash(self) -> int:
        # now we can use hash
        return hash(self.match_set)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ConsumeAny):
            return self.match_set == other.match_set
        if isinstance(other, ConsumeString):
            return (self.match_set.length() == 1
                    and self.match_set.unwrap_value() == other.match_string)
        return NotImplemented

    def __str__(self) -> str:
        return self._symbol() or str(self.match_set)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.match_set!r})"

    def copy(self) -> 'ConsumeAny':
        return ConsumeAny(self.match_set.copy())
        

class MatchConditions:
    _alpha = {chr(i) for i in range(ord('a'), ord('z') + 1)}\
        | {chr(i) for i in range(ord('A'), ord('Z') + 1)}
    _digits = {chr(i) for i in range(ord('0'), ord('9') + 1)}

    _string: str
    _cursor: int

    def __init__(self, string: str) -> None:
        self._string = string
        self._cursor = 0

    # https://en.wikipedia.org/wiki/Nondeterministic_finite_automaton#%CE%B5-closure_of_a_state_or_set_of_states
    @GenericParserPredicate.of(symbol="\u03B5",
                               coverage=SignedSet(negate=True))
    def epsilon_transition(self) -> bool:
        return True

    @ConsumeAny(SignedSet(negate=True))
    def consume_any(self): ...

# region regex tokens
    @represented_by(symbol="d", escaped=True)
    @ConsumeAny(_digits)
    def consume_digit(self): ...

    @represented_by("D", escaped=True)
    @-consume_digit
    def consume_not_digit(self): ...

    @represented_by("w", escaped=True)
    @ConsumeAny(_alpha | _digits | {"_"})
    def consume_alphanum(self): ...

    @represented_by("W", escaped=True)
    @-consume_alphanum
    def consume_not_alphanum(self): ...

    @represented_by("s", escaped=True)
    @ConsumeAny(" \r\n\t\v\f")
    def consume_whitespace(self): ...

    @represented_by("S", escaped=True)
    @-consume_whitespace
    def consume_not_whitespace(self): ...

    @ConsumeAny("\r\n")
    def consume_newline(self): ...

    @represented_by(".")
    @-consume_newline
    def consume_not_newline(self): ...

    @represented_by("Z", escaped=True)
    @GenericParserPredicate.of(symbol="\\Z", coverage=SignedSet())
    def end(self) -> bool:
        return self._cursor >= len(self._string)

    @represented_by("A", escaped=True)
    @GenericParserPredicate.of(symbol="\\A",
                               coverage=SignedSet(negate=True))
    def begin(self):
        return self._cursor <= 0
# endregion regex tokens


CaptureGroup: TypeAlias = int | str
