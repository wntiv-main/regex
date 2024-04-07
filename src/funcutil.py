import functools
from typing import Callable, Concatenate, Optional, ParamSpec, TypeAlias, TypeVar

S = TypeVar("S")
T = TypeVar("T")
T2 = TypeVar("T2")
TArgs = ParamSpec("TArgs")
TArgs2 = ParamSpec("TArgs2")


def optional_chain(func: Callable[[T], T2]) \
        -> Callable[[Optional[T]], Optional[T2]]:
    return lambda x: None if x is None else func(x)


def optional_chain_method(func: Callable[[S, T], T2]) \
        -> Callable[[S, Optional[T]], Optional[T2]]:
    return lambda self, x: None if x is None else func(self, x)


def curry(func: Callable[..., T]) -> Callable[..., Callable[..., T]]:
    return functools.partial(functools.partial, func)


class extend:
    _func_impl: Callable[TArgs2, T]

    def __init__(
            self,
            function_implementation: Callable[..., T],
            *args, **kwargs):
        self._func_impl = functools.partial(
            function_implementation, *args, **kwargs)

    def __call__(self, _: Callable[TArgs2, None])\
            -> Callable[TArgs2, T]:
        return self._func_impl


WrappedFunction: TypeAlias = Callable[TArgs2, T2]
InnerFunction: TypeAlias = Callable[TArgs, T]


def wrap(
        wrapper_function: Callable[Concatenate[InnerFunction, TArgs2], T2],
        *args,
        **kwargs) -> Callable[[InnerFunction], WrappedFunction]:
    def wrap_decorator(inner_function: InnerFunction) -> WrappedFunction:
        return functools.partial(wrapper_function, inner_function,
                                 *args, **kwargs)
    return wrap_decorator


def wrap_method(
        wrapper_function: Callable[Concatenate[InnerFunction, TArgs2], T2],
        *args,
        **kwargs) -> Callable[[InnerFunction], WrappedFunction]:
    def wrap_decorator(inner_function: InnerFunction) -> WrappedFunction:
        return functools.partialmethod(wrapper_function, inner_function,
                                       *args, **kwargs)
    return wrap_decorator


def negate(func: Callable[TArgs, bool]) -> Callable[TArgs, bool]:
    def wrapper(*args, **kwargs):
        return not func(*args, **kwargs)
    return wrapper


def _hash_set(value: set) -> int:
    result = 0
    for element in value:
        result ^= hash(element)
    return result
