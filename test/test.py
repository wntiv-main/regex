from enum import Enum, auto
from traceback import format_exc
from typing import Callable, Self, TypeVarTuple

from .test_error import TestError

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


class TestCase:
    _test_cases: set['TestCase'] = set()

    _callable: Callable[..., None]
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
        results = {test_type: "<tr><th>Test #</th><th>Test</th>"
                              "<th>Expected</th><th>Outcome</th>"
                              "<th>Result</th></tr>"
                   for test_type in TestType}
        counter = 0
        for case in TestCase._test_cases:
            test_name = (case._callable.__name__
                         .replace('_', ' ').capitalize())
            results[case._type] += (
                f"<tr><td>{counter}</td>"
                f"<td>{test_name}: {case._description}</td>"
                f"<td>{_htmlify(case._expected)}</td>"
                f"<td>{_htmlify(case._outcome)}</td>"
                f"<td>{case._result.name}</td></tr>")
            counter += 1
        return "<br/>".join((f"<h1>{test_type.name}</h1><br/>"
                             f"<table>{results[test_type]}</table><br/>"
                             for test_type in TestType))

    def __init__(self, type: TestType, description: str,
                 *, expected: str | None = None):
        self._type = type
        self._description = description
        self._expected = expected
        self._outcome = None
        self._result = None

    def set_expected(self, expected: str) -> None:
        self._expected = expected

    def set_response(self, response: str) -> None:
        self._outcome = response

    # For use as decorator
    def __call__(self, func: Callable[[Self], None]) -> None:
        self._callable = func
        TestCase._test_cases.add(self)

    def _call(self):
        self._callable(self)

    def run(self):
        try:
            self._call()
            self._result = ResultType.PASS
            if self._outcome is None:
                self._outcome = "As expected."
            self.on_success()
        except TestError as e:
            self._result = ResultType.FAIL
            self._outcome = e.outcome_message()
            self.on_fail()
        except Exception as e:
            self._result = ResultType.ERROR
            self._outcome = (f"Test encountered an error: "
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
                f"Outcome: \n{self._outcome}\n"
                f"Result: {self._result.name}")
