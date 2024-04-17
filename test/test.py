from enum import Enum, auto
from traceback import format_exc
from typing import Callable, TypeVarTuple

from .test_error import TestError

TArgs = TypeVarTuple("TArgs")


class ResultType(Enum):
    PASS = auto()
    FAIL = auto()
    ERROR = auto()


class TestCase:
    _test_cases: set['TestCase'] = set()

    _callable: Callable[..., None]
    _description: str
    _expected: str | None
    _response: str | None
    _result: ResultType | None

    @staticmethod
    def run_cases():
        for case in TestCase._test_cases:
            case.run()
            print(case)

    def __init__(self, description: str,
                 *, expected: str | None = None):
        self._description = description
        self._expected = expected
        self._response = None
        self._result = None

    def set_expected(self, expected: str) -> None:
        self._expected = expected

    def set_response(self, response: str) -> None:
        self._response = response

    def __call__(self, func: Callable[..., None]) -> None:
        self._callable = func
        TestCase._test_cases.add(self)

    def _call(self):
        self._callable(self)

    def run(self):
        try:
            self._call()
            self._result = ResultType.PASS
            if self._response is None:
                self._response = "As expected."
            self.on_success()
        except TestError as e:
            self._result = ResultType.FAIL
            self._response = e.outcome_message()
            self.on_fail()
        except Exception as e:
            self._result = ResultType.ERROR
            self._response = (f"Test encountered an error: "
                              f"{format_exc()}")
            self.on_error()

    def on_success(self):
        pass

    def on_fail(self):
        pass

    def on_error(self):
        pass

    def __str__(self) -> str:
        return (f"{self._callable.__name__}: {self._description}\n"
                f"Expected: \n{self._expected}\n"
                f"Response: \n{self._response}\n"
                f"Result: {self._result.name}")
