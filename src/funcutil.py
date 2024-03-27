import functools
from typing import Any, Callable, TypeAlias, TypeVar, TypeVarTuple, Unpack


T = TypeVar("T")
T2 = TypeVar("T2")
TArgs = TypeVarTuple("TArgs")
TArgs2 = TypeVarTuple("TArgs2")


def curry(func: Callable[[Any], T]) -> Callable[[Any], Callable[[Any], T]]:
    return functools.partial(functools.partial, func)


class extend:
    _func_impl: Callable[[*TArgs2], T]

    def __init__(
            self,
            function_implementation: Callable[..., T],
            *args: *TArgs, **kwargs):
        self._func_impl = functools.partial(
            function_implementation, *args, **kwargs)

    def __call__(self, _: Callable[[*TArgs2], None])\
            -> Callable[[*TArgs2], T]:
        return self._func_impl


WrappedFunction: TypeAlias = Callable[[*TArgs2], T2]
InnerFunction: TypeAlias = Callable[[*TArgs], T]


def wrap(
        wrapper_function: Callable[[InnerFunction, *TArgs2], T2],
        *args,
        **kwargs) -> Callable[[InnerFunction], WrappedFunction]:
    def wrap_decorator(inner_function: InnerFunction) -> WrappedFunction:
        return functools.partial(wrapper_function, inner_function,
                                 *args, **kwargs)
    return wrap_decorator


def wrap_method(
        wrapper_function: Callable[[InnerFunction, *TArgs2], T2],
        *args,
        **kwargs) -> Callable[[InnerFunction], WrappedFunction]:
    def wrap_decorator(inner_function: InnerFunction) -> WrappedFunction:
        return functools.partialmethod(wrapper_function, inner_function,
                                       *args, **kwargs)
    return wrap_decorator

# def extend(
#         _: Callable[[*TArgs], None],
#         function_implementation: Callable[[*TArgs], T],
#         *args, **kwargs) -> Callable[[*TArgs], T]:
#     return functools.partialmethod(function_implementation, *args, **kwargs)


def negate(func: Callable[[*TArgs], bool]) -> Callable[[*TArgs], bool]:
    def wrapper(*args, **kwargs):
        return not func(*args, **kwargs)
    return wrapper
