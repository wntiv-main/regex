import functools
from typing import Any, Callable, TypeVar, TypeVarTuple


T = TypeVar("T")


def curry(func: Callable[[Any], T]) -> Callable[[Any], Callable[[Any], T]]:
    return functools.partial(functools.partial, func)


TArgs = TypeVarTuple("TArgs")


def extend(
        _: Callable[[*TArgs], None],
        function_implementation: Callable[[*TArgs], T],
        *args, **kwargs) -> Callable[[*TArgs], T]:
    return functools.partialmethod(function_implementation, *args, **kwargs)


def negate(func: Callable[[*TArgs], bool]) -> Callable[[*TArgs], bool]:
    def wrapper(*args, **kwargs):
        return not func(*args, **kwargs)
    return wrapper
