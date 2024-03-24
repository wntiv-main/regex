import functools
from typing import Any, Callable, TypeVar, TypeVarTuple, Unpack


T = TypeVar("T")
TArgs = TypeVarTuple("TArgs")
TRemaining = TypeVarTuple("TRemaining")


def curry(func: Callable[[Any], T]) -> Callable[[Any], Callable[[Any], T]]:
    return functools.partial(functools.partial, func)


class extend:
    _func_impl: Callable[[*TRemaining], T]

    def __init__(
            self,
            function_implementation: Callable[..., T],
            *args: *TArgs, **kwargs):
        self._func_impl = functools.partial(
            function_implementation, *args, **kwargs)

    def __call__(self, func: Callable[[*TRemaining], None])\
            -> Callable[[*TRemaining], T]:
        return self._func_impl

# def extend(
#         _: Callable[[*TArgs], None],
#         function_implementation: Callable[[*TArgs], T],
#         *args, **kwargs) -> Callable[[*TArgs], T]:
#     return functools.partialmethod(function_implementation, *args, **kwargs)


def negate(func: Callable[[*TArgs], bool]) -> Callable[[*TArgs], bool]:
    def wrapper(*args, **kwargs):
        return not func(*args, **kwargs)
    return wrapper
