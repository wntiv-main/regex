"""Utility classes to help regular expression matching"""

__author__ = "Callum Hynes"
__all__ = ['_parser_symbols', '_parser_symbols_escaped', 'State',
           'SignedSet', 'ParserPredicate', 'GenericParserPredicate',
           'ConsumeAny', 'MatchConditions', 'CaptureGroup']

from typing import (Any, Callable, Generic, Iterable, Optional, Self,
                    TypeAlias, TypeVar, TypeVarTuple, overload,
                    override)
from abc import ABC, abstractmethod
import functools


_parser_symbols: dict[str, 'ParserPredicate'] = {}
"""Symbols that directly correlate to a ParserPredicate"""

_parser_symbols_escaped: dict[str, 'ParserPredicate'] = {}
"""Escaped symbols that directly correlate to a ParserPredicate"""

T = TypeVar("T")
TArgs = TypeVarTuple("TArgs")


State: TypeAlias = int
"""
Represents a single state in the DFA, internally represented by a simple
index into the DFA's state transition table
"""


class SignedSet(Generic[T]):
    """
    A set that can theoretically represent infinite elements, so long as
    either the amount of elements *in* the set, or the amount of
    elements *not in* the set is finite. This allowed descriptions of
    sets where the entire extent of the set is infinite, or not fully
    known. E.g. instead of needing to represent the *entire* unicode
    charset, one would only need to represent the chars either included
    or excluded from the set
    """
    # Lots of internal access to other SignedSet instances
    # pylint: disable=protected-access

    _negate: bool
    """Whether the is negated"""

    _accept: set[T]
    """Black/white-list, depending on value of {_negate}"""

    def __init__(self,
                 value: 'SignedSet[T] | Iterable[T] | None' = None,
                 negate: bool = False):
        if isinstance(value, SignedSet):
            self._accept = value._accept
            self._negate = bool(negate ^ value._negate)
        else:
            self._accept = set() if value is None else set(value)
            self._negate = negate

    def add(self, el: T) -> None:
        """
        Add an element to the set, if it isnt already

        Arguments:
            el -- The element to add
        """
        if self._negate:
            self._accept.discard(el)
        else:
            self._accept.add(el)

    def discard(self, el: T) -> None:
        """
        Remove an element from the set, if it is there

        Arguments:
            el -- The element to remove
        """
        if self._negate:
            self._accept.add(el)
        else:
            self._accept.discard(el)

    def negate(self) -> Self:
        """
        Invert the elements that this set contains

        Returns:
            The current instance
        """
        self._negate = not self._negate
        return self

    def __neg__(self) -> 'SignedSet[T]':
        """
        Find the invert set, which contains ALL elements that are NOT in
        this set

        Returns:
            A new SignedSet, containing all elements not in this set
        """
        return self.copy().negate()

    def __contains__(self, value: T):
        """
        Find if a value is contained within this set

        Arguments:
            value -- The value to check

        Returns:
            Whether the value was in the set
        """
        return self._negate ^ (value in self._accept)

    @overload
    @staticmethod
    def union(*sets: 'SignedSet[T]') -> 'SignedSet[T]':  # type: ignore
        ...

    @overload
    def union(self, *others: 'SignedSet[T]') -> 'SignedSet[T]':
        ...

    # pylint: disable-next=no-self-argument
    def union(*sets: 'SignedSet[T]') -> 'SignedSet[T]':
        """
        Finds the union of the given sets, the set containing all of the
        elements of all the given sets

        Returns:
            A new SignedSet, the union of the other sets
        """
        return SignedSet().i_union(*sets)
    __or__ = union

    def i_union(self, *sets: 'SignedSet[T]') -> Self:
        """
        Adds all the elements from the other set(s) to this set, in
        effect making this the set that contains the elements of all of
        the given sets

        Returns:
            The current instance
        """
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

    @overload
    @staticmethod
    def intersection( # type: ignore
        *sets: 'SignedSet[T]') -> 'SignedSet[T]':
        ...

    @overload
    def intersection(self, *others: 'SignedSet[T]') -> 'SignedSet[T]':
        ...

    # pylint: disable-next=no-self-argument
    def intersection(*sets):  # type: ignore
        """
        Finds the intersection of the given sets, the set of all
        elements that are in ALL the given sets

        Returns:
            A new SignedSet, the intersection of the other sets
        """
        return SignedSet(negate=True).i_intersection(*sets)
    __and__ = intersection

    def i_intersection(self, *sets: 'SignedSet[T]') -> Self:
        """
        Removes all the elements from this set that aren't in all the
        other set(s), in effect making this set the set of all elements
        that are in ALL the given sets

        Returns:
            The current instance
        """
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

    def symmetric_difference(self, other: 'SignedSet[T]') \
            -> 'SignedSet[T]':
        """
        Finds the symmetric difference of the given sets, the set of all
        elements that are in exactly ONE of the two given sets

        Returns:
            A new SignedSet, the symmetric difference of the two sets
        """
        match self._negate, other._negate:
            case (False, False) | (True, True):
                return SignedSet(self._accept ^ other._accept)
            case (False, True) | (True, False):
                return SignedSet(self._accept ^ other._accept, True)
            case _:
                raise ValueError("self._negate or other._negate is not "
                                 "a boolean value")
    __xor__ = symmetric_difference

    def i_symmetric_difference(self, other: 'SignedSet') -> Self:
        """
        Removes all the elements from the other set from this set, and
        adds all the elements of the other set to this set, that aren't
        already in this set. In effect, this set becomes the set of
        all elements that are in exactly ONE of the two original sets

        Returns:
            The current instance
        """
        match self._negate, other._negate:
            case False, False:
                self._accept ^= other._accept
            case True, True:
                self._negate = False
                self._accept ^= other._accept
            case (False, True) | (True, False):
                self._negate = True
                self._accept ^= other._accept
        return self
    __ixor__ = i_symmetric_difference

    def difference(self, other: 'SignedSet[T]') -> 'SignedSet[T]':
        """
        Finds the difference of the given sets, the set of all
        elements in the first set that are NOT in the second set

        Returns:
            A new SignedSet, the difference of the two sets
        """
        match self._negate, other._negate:
            case False, False:
                return SignedSet(self._accept - other._accept)
            case True, True:
                return SignedSet(other._accept - self._accept)
            case False, True:
                return SignedSet(self._accept & other._accept)
            case True, False:
                return SignedSet(self._accept | other._accept, True)
            case _:
                raise ValueError("self._negate or other._negate is not "
                                 "a boolean value")
    __sub__ = difference

    def i_difference(self, other: 'SignedSet') -> Self:
        """
        Removes all the elements from the other set from this set

        Returns:
            The current instance
        """
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

    @staticmethod
    def _hash_set(value: set) -> int:
        """
        Utility function to find the hash of a set

        Arguments:
            value -- The set to be hashed

        Returns:
            The integer hash of the set
        """
        result = 0
        for element in value:
            # XOR is commutative, set order does not matter
            result ^= hash(element)
        return result

    def __hash__(self) -> int:
        """
        Returns:
            The integer hash of the set
        """
        return SignedSet._hash_set(self._accept) ^ (self._negate * ~0)

    def __eq__(self, other: Any) -> bool:
        """
        SignedSet (soft) equality comparison. SignedSets are equal if
        their contents are equal

        Arguments:
            other -- The set to compare against

        Returns:
            Whether or not the objects are equal
        """
        if isinstance(other, SignedSet):
            return (self._negate == other._negate
                    and self._accept == other._accept)
        return NotImplemented

    def __bool__(self):
        """
        Boolean conversion operator

        Returns:
            Whether the set is empty
        """
        return len(self._accept) > 0 or self._negate

    # not using __len__ because sometimes length is not finite
    # (represented by negative, which is not allowed by __len__)
    def length(self) -> int:
        """
        A value representing the "length" of the set

        Returns:
            A positive integer representing the length of the set when
            positive. When negated, a negative integer representing the
            amount of elements excluded from the set
        """
        if self._negate:
            return -len(self._accept)
        return len(self._accept)

    def unwrap_value(self) -> T:
        """
        Gets the one and only value out of the set

        Returns:
            The only value in the set, assuming there is only one value
        """
        assert self.length() == 1
        return self._accept.copy().pop()

    def copy(self) -> 'SignedSet':
        """
        Clones the set

        Returns:
            A new, identical, SignedSet object
        """
        # make sure to copy _____list
        return SignedSet(self._accept.copy(), self._negate)

    def __str__(self):
        """
        Human-readable, compact string representation of the set

        Returns:
            A string representing the set
        """
        return f"{'-' if self._negate else ''}{self._accept}"

    def __repr__(self):
        """
        Human-readable, detailed string representation of the set

        Returns:
            A string representing the set
        """
        return (f"{self.__class__.__name__}"
                f"({self._accept!r}, {self._negate!r})")


class ParserPredicate(ABC):
    """
    Represents a predicate which takes a current state along a string
    buffer, optionally consumes chars from the buffer, and returns
    whether the predicate should match or not
    """

    __doc__: Optional[str] = None
    """Documentation string for this predicate"""

    @abstractmethod
    def evaluate(self, ctx: 'MatchConditions') -> bool:
        """
        Evaluate the predicate on the given string buffer context

        Arguments:
            ctx -- The context

        Returns:
            Whether the predicate should match
        """

    @overload
    def __call__(self, func: Callable) -> Self:
        """
        Overload to allow use as a decorator.

        Returns:
            The current instance in place of the decorated function.
        """

    @overload
    def __call__(self, ctx: 'MatchConditions') -> bool:
        """
        Evaluate the predicate on the given string buffer context

        Arguments:
            ctx -- The context

        Returns:
            Whether the predicate should match
        """

    def __call__(self, *args):
        match args:
            case (x,) if callable(x):
                self.__doc__ = x.__doc__
                return self
            case (MatchConditions() as ctx,):
                return self.evaluate(ctx)
            case _:
                return TypeError(f"Invalid arguments to {self}()")

    @abstractmethod
    def coverage(self) -> SignedSet[str]:
        """
        The range of chars that this predicate *could* match for, used
        to find non-deterministic junctions

        Returns:
            The set of all chars this predicate culd potentially match
        """
        raise NotImplementedError()

    @abstractmethod
    def __hash__(self) -> int:
        """
        The hash of the object, for organising in sets

        Returns:
            The hash of the object
        """
        raise NotImplementedError()

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        """
        Equality comparison

        Arguments:
            other -- object to compare against

        Returns:
            Whether the objects are equal
        """
        raise NotImplementedError()

    def _symbol(self) -> Optional[str]:
        """
        A symbol representing this predicate, if one is available

        Returns:
            A symbol, if one is available, otherwise, None
        """
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
        """
        Human-readable symbolic representation of this predicate

        Returns:
            String representing this predicate
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        """
        Descriptive representation of this predicate

        Returns:
            String representing this predicate
        """
        args_str = ', '.join((f"{key}={value!r}"
                              for key, value in self.__dict__.items()))
        return f"{self.__class__.__name__}({args_str})"

    @abstractmethod
    def copy(self) -> 'ParserPredicate':
        """
        Makes a clone of the current predicate

        Returns:
            A new, identical ParserPredicate instance
        """
        raise NotImplementedError()


# region parser predicates
class GenericParserPredicate(ParserPredicate):
    """
    A simple implementation of ParserPredicate
    """

    _sym: str
    """The symbol to use to represent this predicate"""

    _coverage: SignedSet[str]
    """The set of chars this predicate may cover"""

    _evaluate: Callable[['MatchConditions'], bool]
    """The callable representing the predicate itself"""

    def __init__(self,
                 _symbol: str,
                 _coverage: SignedSet[str],
                 _evaluate: Callable[['MatchConditions'], bool]):
        self._sym = _symbol
        self._coverage = _coverage
        self._evaluate = _evaluate

    @override
    def evaluate(self, ctx: 'MatchConditions') -> bool:
        return self._evaluate(ctx)

    @override
    def coverage(self):
        return self._coverage

    @classmethod
    def of(cls, *, symbol: str, coverage: SignedSet[str]) -> Callable[
            [Callable[['MatchConditions'], bool]],
            'ParserPredicate']:
        """
        Decorator for easy creation of GenericParserPredicates

        Returns:
            A new GenericParserPredicate instance
        """
        return functools.partial(cls, symbol, coverage)

    @override
    def __hash__(self) -> int:
        return id(self._evaluate)

    @override
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, GenericParserPredicate):
            return self._evaluate == other._evaluate
        return NotImplemented

    @override
    def __str__(self) -> str:
        return self._symbol() or self._sym

    @override
    def copy(self) -> 'GenericParserPredicate':
        return GenericParserPredicate(self._sym, self._coverage,
                                      self._evaluate)


class ConsumeAny(ParserPredicate):
    """Predicate which matches and consumes any char in a set"""

    match_set: SignedSet[str]
    """The set of strings that this predicate matches for"""

    def __init__(self, match_set: SignedSet[str] | Iterable[str]) -> None:
        self.match_set = SignedSet(match_set)

    @override
    def evaluate(self, ctx: 'MatchConditions') -> bool:
        # pylint: disable=protected-access
        if (not ctx.end(ctx) and
                ctx._string[ctx._cursor] in self.match_set):
            ctx._cursor += 1
            return True
        return False

    @override
    def coverage(self):
        return self.match_set

    def __neg__(self) -> 'ConsumeAny':
        """
        Get the predicate which matches the opposite chars

        Returns:
            A new ConsumeAny predicate, matching only the chars that
            this does NOT match.
        """
        return ConsumeAny(-self.match_set)

    @override
    def __hash__(self) -> int:
        return hash(self.match_set)

    @override
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ConsumeAny):
            return self.match_set == other.match_set
        return NotImplemented

    @override
    def __str__(self) -> str:
        return self._symbol() or str(self.match_set)

    @override
    def copy(self) -> 'ConsumeAny':
        return ConsumeAny(self.match_set.copy())
# endregion parser predicates


P = TypeVar("P", bound=ParserPredicate)


class _RepresentedBy: # pylint: disable=too-few-public-methods
    """Simple decorator to populate _parser_symbols(_escaped)"""

    _symbol: str
    """The symbol that represents the predicate"""

    _escaped: bool
    """Whether the symbol should be escaped"""

    def __init__(self, symbol: str, *, escaped: bool = False):
        self._symbol = symbol
        self._escaped = escaped

    def __call__(self, func: P) -> P:
        """
        Adds the given function to the map of parser symbols

        Arguments:
            func -- The predicate to add

        Returns:
            The original predicate
        """
        if self._escaped:
            _parser_symbols_escaped[self._symbol] = func
        else:
            _parser_symbols[self._symbol] = func
        return func


class MatchConditions:
    """
    A string buffer with methods for conditional consumption of the
    string
    """

    _alpha = {chr(i) for i in range(ord('a'), ord('z') + 1)}\
        | {chr(i) for i in range(ord('A'), ord('Z') + 1)}
    """Set of alphabetical chars"""
    _digits = {chr(i) for i in range(ord('0'), ord('9') + 1)}
    """Set of digit chars"""

    _string: str
    """The string "buffer" currently matching against"""
    _cursor: int
    """The current index into the string buffer"""

    def __init__(self, string: str) -> None:
        self._string = string
        self._cursor = 0

    # https://en.wikipedia.org/wiki/Nondeterministic_finite_automaton#%CE%B5-closure_of_a_state_or_set_of_states
    @GenericParserPredicate.of(
        symbol="\u03B5", coverage=SignedSet(negate=True)) # type: ignore
    def epsilon_transition(self) -> bool:
        """Always matches without consuming any chars"""
        return True

    @ConsumeAny(SignedSet(negate=True))
    def consume_any(self):
        """Always matches, and consumes a char"""

# region regex tokens
# Pylint fails to recognise __neg__ defined on ConsumeAny instances
# pylint: disable=invalid-unary-operand-type
    @_RepresentedBy(symbol="d", escaped=True)
    @ConsumeAny(_digits)
    def consume_digit(self):
        """Matches and consumes any digit char"""

    @_RepresentedBy("D", escaped=True)
    @-consume_digit
    def consume_not_digit(self):
        """Matches and consumes any non-digit char"""

    @_RepresentedBy("w", escaped=True)
    @ConsumeAny(_alpha | _digits | {"_"})
    def consume_alphanum(self):
        """Matches and consumes any alphanumeric char, including '_'"""

    @_RepresentedBy("W", escaped=True)
    @-consume_alphanum
    def consume_not_alphanum(self):
        """Matches and consumes any non-alphanumeric char"""

    @_RepresentedBy("s", escaped=True)
    @ConsumeAny(" \r\n\t\v\f")
    def consume_whitespace(self):
        """Matches and consumes any whitespace char"""

    @_RepresentedBy("S", escaped=True)
    @-consume_whitespace
    def consume_not_whitespace(self):
        """Matches and consumes any non-whitespace char"""

    @ConsumeAny("\r\n")
    def consume_newline(self):
        """Matches and consumes any newline char"""

    @_RepresentedBy(".")
    @-consume_newline
    def consume_not_newline(self):
        """Matches and consumes any char, excluding newlines"""

    @_RepresentedBy("Z", escaped=True)
    @GenericParserPredicate.of(symbol="\\Z",
                               coverage=SignedSet()) # type: ignore
    def end(self) -> bool:
        """Matches the end of the buffer"""
        return self._cursor >= len(self._string)

    @_RepresentedBy("A", escaped=True)
    @GenericParserPredicate.of(
        symbol="\\A", coverage=SignedSet(negate=True)) # type: ignore
    def begin(self):
        """Matches the beginning of the buffer"""
        return self._cursor <= 0
# endregion regex tokens


CaptureGroup: TypeAlias = int | str
"""Represents the descriptor of a capture group"""
