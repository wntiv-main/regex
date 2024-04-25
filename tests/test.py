"""Utilities for running unit tests"""

__author__ = "Callum Hynes"
__all__ = ["ResultType", "TestType", "TestCase", "_copy_html",
           "AssertRaises", "AssertNoRaises"]

from abc import ABC, abstractmethod
from enum import Enum, auto
from traceback import format_exc
from typing import Any, Callable, TypeVarTuple, override

from .test_error import ExceptionAsTestError, TestError, TestErrorImpl


TArgs = TypeVarTuple("TArgs")


class ResultType(Enum):
    """Represents the outcome of the test"""
    PASS = auto()
    FAIL = auto()
    ERROR = auto()
    TOTAL = auto()  # Used only for printing


class TestType(Enum):
    """Represents the A/M/E criterion the test was aimed toward"""
    EXPECTED = auto()
    BOUNDARY = auto()
    INVALID = auto()
    TOTAL = auto()  # Used only for printing


def _replace_markdown_with_tag(
        source: str,
        markdown_seq: str, tag_name: str) -> str:
    """
    Replaces markdown-style formatting in {source} with the HTML
    equivilent

    Arguments:
        source -- The string to search and replace in
        markdown_seq -- The markdown decorator to search for (e.g. *)
        tag_name -- The HTML tag to replace it with

    Returns:
        The refactored string
    """
    while (idx := source.find(markdown_seq)) != -1:
        if (source.find(markdown_seq,
                        idx + len(markdown_seq) + 1)) != -1:
            source = (source.replace(markdown_seq, f"<{tag_name}>", 1)
                      .replace(markdown_seq, f"</{tag_name}>", 1))
        else:
            break
    return source


def _htmlify(content: Any, *, _TAB='\t') -> str:
    """Formats the content to be printed as HTML"""
    result_lines = str(content).splitlines(keepends=True)
    current_indent = -1
    # Handle list structures
    for i, line in enumerate(result_lines):
        indent = 0
        while line.startswith(_TAB):
            line = line[len(_TAB):]
            indent += 1
        if line.startswith("- "):
            line = f"<li>{line.removeprefix("- ").strip()}</li>"
            if indent > current_indent:
                line = f"<ul>{line}"
        else:
            indent -= 1
        if indent < current_indent:
            line = f"</ul>{line}"
        current_indent = indent
        result_lines[i] = line
    result = ''.join(result_lines)
    result += "</ul>" * (current_indent + 1)  # close remaining tags

    result = result.replace('\n<ul>', '<ul>')
    result = _replace_markdown_with_tag(result, "```", "pre")
    result = result.replace('<pre>\n', '<pre>')
    result = result.replace('\n</pre>', '</pre>')
    result = _replace_markdown_with_tag(result, "`", "code")
    result = result.replace('\n', '<br/>')
    return result


def _copy_html(  # pylint: disable=dangerous-default-value
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
        _CF_HTML=[]) -> bool:
    """
    Cursed function to copy HTML content to the user's clipboard

    Arguments:
        content -- The HTML content to copy.

    Returns:
        Whether the contents were successfully copied
    """
    # Adapted from:
    # https://stackoverflow.com/questions/55698762/how-to-copy-html-code-to-clipboard-using-python
    try:
        # pylint: disable-next=import-outside-toplevel
        import win32clipboard
    except ImportError:
        print("WARNING: `pip install pywin32` is needed to copy "
              "HTML output.")
        return False
    # pylint: disable=c-extension-no-member
    if not _CF_HTML:
        _CF_HTML.append(win32clipboard
                        .RegisterClipboardFormat("HTML Format"))
    html = _HTML_PREFIX + content + _HTML_SUFFIX
    frag_start = len(_HTML_PREFIX)
    frag_end = len(_HTML_PREFIX) + len(content)
    try:
        win32clipboard.OpenClipboard(0)
        win32clipboard.EmptyClipboard()
        # How long is the prefix going to be?
        dummy_prefix = _MARKER % (0, 0, 0, 0, 0, 0, "file://null")
        len_prefix = len(dummy_prefix)
        prefix = _MARKER % (len_prefix, len(html)+len_prefix,
                            frag_start+len_prefix, frag_end+len_prefix,
                            frag_start+len_prefix, frag_end+len_prefix,
                            "file://null")
        src = (prefix + html).encode("UTF-8")
        # print(src)
        win32clipboard.SetClipboardData(_CF_HTML[0], src)
        return True
    finally:
        win32clipboard.CloseClipboard()


class TestCase(ABC):
    """Represents a test to be ran on the program"""

    _test_cases: list['TestCase'] = []
    """List of all tests which have been created"""

    _type: TestType
    """The A/M/E criterion this test was aimed for"""

    _description: str
    """A description of what is being tested"""

    _expected: str | None
    """A description of the expected outcome of this test"""

    _outcome: str | None
    """A description of the actual outcome of this test"""

    _result: ResultType | None
    """The overall result of the test"""

    @staticmethod
    def run_cases() -> dict[TestType, dict[ResultType, int]]:
        """
        Runs all of the test cases that have been declared thus far

        Returns:
            Table of the amount of tests, by test result and A/M/E
                criterion
        """
        # None represents total
        result = {
            test: {result: 0
                   for result in ResultType}
            for test in TestType
        }
        for case in TestCase._test_cases:
            # pylint: disable=protected-access
            case.run()
            assert case._result is not None
            # Update category and totals
            result[case._type][case._result] += 1
            result[case._type][ResultType.TOTAL] += 1
            result[TestType.TOTAL][case._result] += 1
            result[TestType.TOTAL][ResultType.TOTAL] += 1
        return result

    @staticmethod
    def format_results_table(
            results: dict[TestType,
                          dict[ResultType, int]],
            *, _col_width: int = 12) -> str:
        """
        Formats the resultant table from TestCase.run_cases() for
        human-readable printing

        Arguments:
            results -- The results to be formatted

        Keyword Arguments:
            _col_width -- The width of the columns in the table
                (default: {12})

        Returns:
            String containing the output table
        """
        out = " " * _col_width
        for header in TestType:
            out += f"{header.name:^{_col_width}}"
        for row in ResultType:
            out += f"\n{row.name:>{_col_width}}"
            for column in TestType:
                out += f"{results[column][row]:>{_col_width}}"
        return out

    @staticmethod
    def produce_html_printout() -> str:
        """
        Produces a detailed HTML table containing descriptions and
        results of each test

        Returns:
            The output HTML as a string
        """
        td_style = "border: 1px solid black;"
        results = {test_type: (
            "<tr><th>Test #</th><th>Test</th>"
            "<th>Expected</th><th>Outcome</th>"
            "<th>Result</th></tr>").replace('<th>',
                                            f'<th style="{td_style}">')
            for test_type in TestType if test_type != TestType.TOTAL}
        counters = {test_type: 0 for test_type in TestType
                    if test_type != TestType.TOTAL}
        for case in TestCase._test_cases:
            # pylint: disable=protected-access
            assert case._result is not None
            results[case._type] += (
                f"<tr><td>{counters[case._type]}</td>"
                f"<td>{_htmlify(case._description)}</td>"
                f"<td>{_htmlify(case._expected)}</td>"
                f"<td>{_htmlify(case._outcome)}</td>"
                f"<td>{case._result.name}</td></tr>").replace(
                    '<td>', f'<td style="{td_style}">')
            counters[case._type] += 1
        table_style = "border-collapse: collapse;"
        return "<br/>".join((f"<h1>{test_type.name}</h1><br/>"
                             f"<table style=\"{table_style}\">"
                             f"{results[test_type]}</table><br/>"
                             for test_type in TestType
                             if test_type != TestType.TOTAL))

    def __init__(self, test_type: TestType, description: str,
                 *, expected: str | None = None):
        self._type = test_type
        self._description = description
        self._expected = expected
        self._outcome = None
        self._result = None
        TestCase._test_cases.append(self)

    def set_expected(self, expected: str) -> None:
        """Sets the expected outcome description"""
        self._expected = expected

    def set_outcome(self, response: str) -> None:
        """Sets the actual outcome description"""
        self._outcome = response

    @abstractmethod
    def _inner_test(self) -> None:
        """
        Method which is actually tested. Should be overriden by test
        cases to implement the test to run

        Raises:
            TestError: When the test should fail
        """
        raise NotImplementedError()

    def run(self) -> None:
        """Runs the test, and populates the result fields"""
        try:
            self._inner_test()
            self._result = ResultType.PASS
            if self._outcome is None:
                self._outcome = "As expected."
        except TestError as e:
            self._result = ResultType.FAIL
            self._outcome = e.outcome_message()
        except Exception:  # pylint: disable=broad-exception-caught
            self._result = ResultType.ERROR
            self._outcome = (f"Test encountered an error: "
                             f"{format_exc()}")

    def __str__(self) -> str:
        """Simple string representation fdor printing"""
        assert self._result is not None
        return (f"--- {self._description} ---\n"
                f"Expected: \n{self._expected}\n"
                f"Outcome: \n{self._outcome}\n"
                f"Result: {self._result.name}")


class AssertRaises(TestCase):
    """Test case for testing code which *should* raise an exception"""

    _exc_type: type[Exception]
    """The type of the exception that should be raised"""

    _callable: Callable[..., None]
    """The inner function to test"""

    def __init__(self, exc_type: type[Exception], description: str,
                 test_type: TestType = TestType.INVALID):
        super().__init__(
            test_type, description,
            expected=f"Expected {exc_type.__name__} to be raised.")
        self._exc_type = exc_type

    # For use as decorator
    def __call__(self, func: Callable[..., None]) -> None:
        self._callable = func
        self._description = (f"{func.__name__.replace('_', ' ')}: "
                             f"{self._description}")

    @override
    def _inner_test(self, *args, **kwargs) -> None:
        """
        Runs the test function

        Raises:
            TestErrorImpl: If the expected exception was not raised
        """
        try:
            self._callable(*args, **kwargs)
            raise TestErrorImpl(f"{self._exc_type.__name__} was not "
                                f"raised.")
        except self._exc_type as e:
            self.set_outcome(f"Exception was raised: ```\n{e}\n```")


class AssertNoRaises(TestCase):
    """
    Test case for testing code which should not raise a given type of
    exception
    """

    _exc_type: type[Exception]
    """The type of the exception that should not be raised"""

    _callable: Callable[..., None]
    """The inner function to test"""

    def __init__(self, exc_type: type[Exception], description: str,
                 test_type: TestType = TestType.INVALID):
        super().__init__(
            test_type, description,
            expected=f"Expected `{exc_type.__name__}` not to be raised")
        self._exc_type = exc_type

    # For use as decorator
    def __call__(self, func: Callable[..., None]) -> None:
        self._callable = func
        self._description = (f"{func.__name__.replace('_', ' ')}: "
                             f"{self._description}")

    @override
    def _inner_test(self, *args, **kwargs) -> None:
        """
        Runs the test function

        Raises:
            ExceptionAsTestError: As a wrapper for the expected
                exception type in case it is raised
        """
        try:
            self._callable(*args, **kwargs)
        except self._exc_type as e:
            raise ExceptionAsTestError() from e
