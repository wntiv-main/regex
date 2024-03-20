from abc import ABC, abstractmethod
import functools
from typing import Any, Callable, TypeVar, TypeVarTuple

from regex import RegexParser

# TODO: https://www.geeksforgeeks.org/visualize-graphs-in-python/

T = TypeVar("T")
def curry(func: Callable[[Any], T]) -> Callable[[Any], Callable[[Any], T]]:
    return functools.partial(functools.partial, func)

# Example use:
# @curry
# def add(a, b):
#     return a + b

# add(1)(2)
class ParserPredicate(Callable[['RegexParser'], bool]):
    _callable: Callable[['RegexParser'], bool]
    
    def __init__(self, function: Callable[['RegexParser'], bool]):
        self._callable = function

    def __call__(self, ctx: 'RegexParser') -> bool:
        return self._callable(ctx)
    
    def compile(self):
        pass

@staticmethod
def represented_by(func: ParserPredicate, symbol: str):
    RegexParser.parser_symbols[symbol] = func
    return func

class RegexParser:
    parser_symbols: dict[str, 'ParserPredicate'] = {}
    _string: str
    _cursor: int
    
    def try_consume(self, match_string: str) -> bool:
        if self._string[
            self._cursor
            : self._cursor + len(match_string)] == match_string:
            self._cursor += len(match_string)
            return True
        return False
    
    def try_consume_any(self, s: set[str]) -> bool:
        if self._string[self._cursor : self._cursor + 1] in s:
            self._cursor += 1
            return True
        return False
    
    @represented_by("$")
    @ParserPredicate
    def end(self) -> bool:
        return self._cursor >= len(self._string)

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

    predicate: AbstractRegexParser.ParserPredicate