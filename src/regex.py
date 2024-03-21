from abc import ABC, abstractmethod
import functools
from typing import Any, Callable, TypeVar, TypeVarTuple

from regex import RegexParser

# TODO: https://www.geeksforgeeks.org/visualize-graphs-in-python/

T = TypeVar("T")
def curry(func: Callable[[Any], T]) -> Callable[[Any], Callable[[Any], T]]:
    return functools.partial(functools.partial, func)

TArgs = TypeVarTuple("TArgs")
def extend(_: Callable[[*TArgs], None], function_implementation: Callable[[*TArgs], T], *args, **kwargs) -> Callable[[*TArgs], T]:
    return functools.partialmethod(function_implementation, *args, **kwargs)

def bool_not(func: Callable[[*TArgs], bool]) -> Callable[[*TArgs], bool]:
    def wrapper(*args, **kwargs):
        return not func(*args, **kwargs)
    return wrapper

# Example use:
# @curry
# def add(a, b):
#     return a + b

# add(1)(2)
# class ParserPredicate(Callable[['RegexParser'], bool]):
#     _callable: Callable[['RegexParser'], bool]
    
#     def __init__(self, function: Callable[['RegexParser'], bool]):
#         self._callable = function

#     def __call__(self, ctx: 'RegexParser') -> bool:
#         return self._callable(ctx)
    
#     def compile(self):
#         pass

def represented_by(func: Callable[[RegexParser], bool], symbol: str, *, escaped: bool = False):
    RegexParser.parser_symbols[escaped][symbol] = func
    return func

class RegexParser:
    _alpha = {chr(i) for i in range(ord('a'), ord('z') + 1)}
    _digits = {chr(i) for i in range(ord('0'), ord('9') + 1)}
    parser_symbols: dict[bool, dict[str, Callable[[RegexParser], bool]]] = {}
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
    
    @represented_by("d", escaped=True)
    @extend(try_consume_any, _digits)
    def consume_digit(self): pass

    @represented_by("D", escaped=True)
    @bool_not
    @extend(consume_digit)
    def consume_not_digit(self): pass

    @represented_by("w", escaped=True)
    @extend(try_consume_any, _alpha + _digits + {"_"})
    def consume_alphanum(self): pass

    @represented_by("W", escaped=True)
    @bool_not
    @extend(consume_alphanum)
    def consume_not_alphanum(self): pass

    @represented_by("s", escaped=True)
    @extend(try_consume_any, set(" \r\n\t\v\f"))
    def consume_whitespace(self): pass

    @represented_by("S", escaped=True)
    @bool_not
    @extend(consume_whitespace)
    def consume_not_whitespace(self): pass

    @extend(try_consume_any, set("\r\n"))
    def consume_newline(self): pass

    @represented_by(".", escaped=True)
    @bool_not
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



class CodeBuilder:
    pass


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
    next: list["Path"]
    previous: list["Path"]

class Path(ABC):
    next: State
    previous: State

    opens: list[CaptureGroup]
    closes: list[CaptureGroup]

    predicate: Callable[[RegexParser], bool]
