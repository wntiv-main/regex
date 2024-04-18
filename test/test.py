from abc import ABC, abstractmethod
from enum import Enum, auto
from traceback import format_exc
from typing import Callable, TypeVarTuple

from .test_error import TestError, TestErrorImpl

TArgs = TypeVarTuple("TArgs")


class ResultType(Enum):
    PASS = auto()
    FAIL = auto()
    ERROR = auto()


class TestType(Enum):
    PASSING = auto()
    BOUNDARY = auto()
    ERROR = auto()


def _htmlify(content: str):
    return (content.replace('\n', '<br/>'))


def _copy_html(
        content: str, *,
        _HTML_PREFIX="<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0 "
        "Transitional//EN\"><HTML><HEAD></HEAD><BODY>"
        "<!--StartFragment-->",
        _HTML_SUFFIX="<!--EndFragment--></BODY></HTML>",
        _MARKER="Version:1.0\r\n"
    "StartHTML:%09d\r\n"
    "EndHTML:%09d\r\n"
    "StartFragment:%09d\r\n"
    "EndFragment:%09d\r\n"
    "StartSelection:%09d\r\n"
    "EndSelection:%09d\r\n"
    "SourceURL:%s\r\n",
        _CF_HTML=[]):
    # Adapted from https://stackoverflow.com/questions/55698762/how-to-copy-html-code-to-clipboard-using-python
    """
    Cursed function to copy HTML content to the user's clipboard

    Arguments:
        content -- The HTML content to copy.
    """
    try:
        import win32clipboard
    except ImportError:
        print("WARNING: `pip install pywin32` is needed to copy "
              "HTML output.")
        return
    if not _CF_HTML:
        _CF_HTML.append(win32clipboard
                        .RegisterClipboardFormat("HTML Format"))
    html = _HTML_PREFIX + content + _HTML_SUFFIX
    fragStart = len(_HTML_PREFIX)
    fragEnd = len(_HTML_PREFIX) + len(content)
    try:
        win32clipboard.OpenClipboard(0)
        win32clipboard.EmptyClipboard()
        # How long is the prefix going to be?
        dummyPrefix = _MARKER % (0, 0, 0, 0, 0, 0, "file://null")
        lenPrefix = len(dummyPrefix)
        prefix = _MARKER % (lenPrefix, len(html)+lenPrefix,
                            fragStart+lenPrefix, fragEnd+lenPrefix,
                            fragStart+lenPrefix, fragEnd+lenPrefix,
                            "file://null")
        src = (prefix + html).encode("UTF-8")
        # print(src)
        win32clipboard.SetClipboardData(_CF_HTML[0], src)
    finally:
        win32clipboard.CloseClipboard()


class TestCase(ABC):
    _test_cases: list['TestCase'] = list()

    _type: TestType
    _description: str
    _expected: str | None
    _outcome: str | None
    _result: ResultType | None

    @staticmethod
    def run_cases():
        for case in TestCase._test_cases:
            case.run()

    @staticmethod
    def produce_html_printout() -> str:
        td_style = "border: 1px solid black;"
        results = {test_type: ("<tr><th>Test #</th><th>Test</th>"
                              "<th>Expected</th><th>Outcome</th>"
                               "<th>Result</th></tr>").replace(
            '<th>', f'<th style="{td_style}">')
                   for test_type in TestType}
        counters = {test_type: 0 for test_type in TestType}
        for case in TestCase._test_cases:
            results[case._type] += (
                f"<tr><td>{counters[case._type]}</td>"
                f"<td>{case._description}</td>"
                f"<td><pre>{_htmlify(case._expected)}</pre></td>"
                f"<td><pre>{_htmlify(case._outcome)}</pre></td>"
                f"<td>{case._result.name}</td></tr>").replace(
                    '<td>', f'<td style="{td_style}">')
            counters[case._type] += 1
        table_style = "border-collapse: collapse;"
        return "<br/>".join((f"<h1>{test_type.name}</h1><br/>"
                             f"<table style=\"{table_style}\">"
                             f"{results[test_type]}</table><br/>"
                             for test_type in TestType))

    def __init__(self, test_type: TestType, description: str,
                 *, expected: str | None = None):
        self._type = test_type
        self._description = description
        self._expected = expected
        self._outcome = None
        self._result = None
        TestCase._test_cases.append(self)

    def set_expected(self, expected: str) -> None:
        self._expected = expected

    def set_outcome(self, response: str) -> None:
        self._outcome = response

    @abstractmethod
    def _inner_test(self) -> None:
        raise NotImplementedError()

    def run(self):
        try:
            self._inner_test()
            self._result = ResultType.PASS
            if self._outcome is None:
                self._outcome = "As expected."
        except TestError as e:
            self._result = ResultType.FAIL
            self._outcome = e.outcome_message()
        except Exception as e:
            self._result = ResultType.ERROR
            self._outcome = (f"Test encountered an error: "
                             f"{format_exc()}")

    def __str__(self) -> str:
        return (f"{self._description}\n"
                f"Expected: \n{self._expected}\n"
                f"Outcome: \n{self._outcome}\n"
                f"Result: {self._result.name}")


class AssertRaises(TestCase):
    _exc_type: type[Exception]
    _callable: Callable[..., None]

    def __init__(self, exc_type: type[Exception], description: str,
                 test_type: TestType = TestType.ERROR):
        super().__init__(
            test_type, description,
            expected=f"Expected {exc_type.__name__} to be raised.")
        self._exc_type = exc_type

    # For use as decorator
    def __call__(self, func: Callable[..., None]) -> None:
        self._callable = func
        self._description = (f"{func.__name__.replace('_', ' ')}: "
                             f"{self._description}")

    def _inner_test(self, *args, **kwargs) -> None:
        try:
            self._callable(*args, **kwargs)
            raise TestErrorImpl(f"{self._exc_type.__name__} was not "
                                f"raised.")
        except self._exc_type as e:
            self.set_outcome(f"Exception was raised: {e}.")
