r"""
Regular expression library using deterministic finite automaton (DFA)
to build and evlauate regular expressions efficiently.

Usage:
    rx = Regex(r"\(\d+(?:,\s*\d+))\)") # match integer tuple e.g. (1, 2)
    rx.test("(1, 3, 37,14)") # True
"""

__author__ = "Callum Hynes"

from src import *

if __name__ == "__main__":
    import test
    test.run_tests()
