import functools
from typing import Any, Callable, Self, TypeAlias, TypeVar, TypeVarTuple


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


class Mutable:
    _instance: 'Mutable' = None
    __slots__ = ()

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


class _UnsafeMutable(type):
    @staticmethod
    def _property(name: str) -> property:
        @property
        def prop(self):
            return getattr(self, name)

        @prop.setter
        def prop(self, value):
            if self._UnsafeMutable__currently_mutable:
                setattr(self, name, value)
            else:
                raise RuntimeError(
                    f"{self.__class__.__name__} mutated unsafely: "
                    f"tried to modify {name}")
        return prop

    def __new__(cls, clsname: str, bases: tuple[type], attrs: dict):
        for name, value in attrs.copy().items():
            if not name.startswith('_') or value is not Mutable():
                continue

            attrs[name[1:]] = _UnsafeMutable._property(name)

        return super().__new__(cls, clsname, bases, attrs)


class UnsafeMutable(metaclass=_UnsafeMutable):
    __currently_mutable: bool

    def __init__(self) -> None:
        self.__currently_mutable = False

    @staticmethod
    def mutator(func: Callable[['UnsafeMutable', *TArgs], T]) \
            -> Callable[['UnsafeMutable', *TArgs], T]:
        def wrapper(self: UnsafeMutable, *args: *TArgs, **kwargs) -> T:
            if self.__currently_mutable:
                return func(self, *args, **kwargs)
            else:
                raise RuntimeError(
                    f"{self.__class__.__name__} mutated unsafely: "
                    f"tried to use mutator {func.__name__}")
        return wrapper

    def __enter__(self) -> Self:
        if self.__currently_mutable:
            raise RuntimeError(f"{self} is already being mutated")
        self.__currently_mutable = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.__currently_mutable = False
        return False
