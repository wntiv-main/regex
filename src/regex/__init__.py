r"""
Regular expression library using deterministic finite automaton (DFA)
to build and evaluate regular expressions efficiently

Usage:
    rx = Regex(r"\(\d+(?:,\s*\d+))\)") # match integer tuple e.g. (1, 2)
    rx.test("(1,3, 37,14)") # True
"""

__author__ = "Callum Hynes"
__all__ = ["Regex"]
__version__ = "0.0.1"

from .regex import Regex
