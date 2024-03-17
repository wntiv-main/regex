from abc import ABC, abstractmethod
from typing import Any, Callable

# TODO: https://www.geeksforgeeks.org/visualize-graphs-in-python/

class ParserPredicate(Callable[['StringProvider'], bool]):
    _callable: Callable[['StringProvider'], bool]
    _compiled: str
    
    def __init__(self, function: Callable[['StringProvider'], bool], compiled: str):
        self._callable = function
        self._compiled = compiled
    
    def __call__(self, string: 'StringProvider') -> bool:
        return self._callable

class StringProvider:
    _string: str
    _cursor: int

    @ParserPredicate()
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
    
    def end(self) -> bool:
        return self._cursor >= len(self._string)
    
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

    predicate: 