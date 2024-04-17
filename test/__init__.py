from src.regexutil import MatchConditions
from .regex_matcher import NodeMatcher, RegexState, assert_regex
from .test import TestCase


@assert_regex(r"a*")
def test_star(start: NodeMatcher):
    start.is_also(RegexState.END).has_literal('a', RegexState.SELF)


@assert_regex(r"a+")
def test_plus(start: NodeMatcher):
    end = start.has_literal('a', RegexState.END)
    end.has_literal('a', RegexState.SELF)


@assert_regex(r"a?")
def test_optional(start: NodeMatcher):
    start.has_literal('a', RegexState.END)
    start.has(MatchConditions.epsilon_transition, RegexState.END)


@assert_regex(r"a|b")
def test_or(start: NodeMatcher):
    start.has_literal('a', RegexState.END)
    start.has_literal('b', RegexState.END)


@assert_regex(r"[ab]")
def test_class(start: NodeMatcher):
    start.has_any('ab', RegexState.END)


@assert_regex(r"[^ab]")
def test_not_class(start: NodeMatcher):
    start.has_any_except('ab', RegexState.END)


def run_tests():
    TestCase.run_cases()
    TestCase.copy_html(TestCase.produce_html_printout())
    assert_regex._failed_regex.display()
