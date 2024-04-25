"""Run tests"""

__author__ = "Callum Hynes"

import sys

# pylint: disable-next=wildcard-import, unused-import
from . import *  # Configure test suite
from .test import TestCase, ResultType, _copy_html
from .regex_tests import TestRegexShape


print("Running tests...")
results = TestCase.run_cases()
print("TEST SUMMARY:")
print(TestCase.format_results_table(results))
if "--full-output" in sys.argv:
    # pylint: disable-next=invalid-name
    output = TestCase.produce_html_printout()
    if _copy_html(output):
        print("Full results in clipboard")
    else:
        # Print header
        print(f"\n{'='*20}\n|{'Full Output':^18}|\n{'='*20}\n")
        print(output)
failed = sum(tests[ResultType.FAIL] + tests[ResultType.ERROR]
             for tests in results.values())
if failed:
    if __debug__ and not '--headless' in sys.argv:
        # pylint: disable-next=protected-access
        TestRegexShape._failed_regex.display()
    else:
        TestCase.print_failed()
sys.exit(failed)
