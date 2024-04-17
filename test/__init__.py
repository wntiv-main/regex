from src.regexutil import MatchConditions
from .regex_matcher import NodeMatcher, RegexState, TestRegexMatches, TestRegexShape
from .test import TestCase, TestType, _copy_html


# Star
@TestRegexShape(r"a*")
def test_star(start: NodeMatcher):
    start.is_also(RegexState.END).has_literal('a', RegexState.SELF)


@TestRegexMatches(r"a*\Z")
def test_star_matches(test: TestRegexMatches):
    test.assert_matches('', 'a', 'aaa', 'aaaaaa')
    test.assert_doesnt_match('njds', 'hello')


@TestRegexMatches(r"a*", TestType.BOUNDARY)
def test_star_matches(test: TestRegexMatches):
    test.assert_matches('')


# Plus
@TestRegexShape(r"a+")
def test_plus(start: NodeMatcher):
    end = start.has_literal('a', RegexState.END)
    end.has_literal('a', RegexState.SELF)


@TestRegexMatches(r"a+")
def test_plus_matches(test: TestRegexMatches):
    test.assert_matches('a', 'aaa', 'aaaaaa', 'aaaaaabdsjmn')
    test.assert_doesnt_match('', 'njds', 'hello')


@TestRegexMatches(r"a+", TestType.BOUNDARY)
def test_plus_matches(test: TestRegexMatches):
    test.assert_matches('a')
    test.assert_doesnt_match('')


# Optional
@TestRegexShape(r"a?")
def test_optional(start: NodeMatcher):
    start.has_literal('a', RegexState.END)
    start.has(MatchConditions.epsilon_transition, RegexState.END)


@TestRegexMatches(r"a?\Z")
def test_optional_matches(test: TestRegexMatches):
    test.assert_matches('', 'a')
    test.assert_doesnt_match('aaa', 'aaaaaaa')


# OR
@TestRegexShape(r"a|b")
def test_or(start: NodeMatcher):
    start.has_literal('a', RegexState.END)
    start.has_literal('b', RegexState.END)


@TestRegexMatches(r"a|b")
def test_or_matches(test: TestRegexMatches):
    test.assert_matches('a', 'b')
    test.assert_doesnt_match('c', 'd')


@TestRegexShape(r"hello|goodbye")
def test_longer_or(start: NodeMatcher):
    start.has_literal_chain('hello', RegexState.END)
    start.has_literal_chain('goodbye', RegexState.END)


@TestRegexMatches(r"hello\Z|goodbye\Z")
def test_longer_or_matches(test: TestRegexMatches):
    test.assert_matches('hello', 'goodbye')
    test.assert_doesnt_match('helloodbye', 'he', 'hellodbye')


# Char classes
@TestRegexShape(r"[ab]")
def test_class_specifier(start: NodeMatcher):
    start.has_any('ab', RegexState.END)


@TestRegexMatches(r"[ab]")
def test_class_specifier_matches(test: TestRegexMatches):
    test.assert_matches('a', 'b')
    test.assert_doesnt_match('c', 'd')


@TestRegexShape(r"[^ab]")
def test_inverted_class_specifier(start: NodeMatcher):
    start.has_any_except('ab', RegexState.END)


@TestRegexMatches(r"[^ab]")
def test_inverted_class_specifier_matches(test: TestRegexMatches):
    test.assert_matches('c', 'd')
    test.assert_doesnt_match('a', 'b')


# Quantifiers
@TestRegexShape(r"a{3}")
def test_quantifier(start: NodeMatcher):
    start.has_literal_chain('aaa', RegexState.END)


@TestRegexMatches(r"a{3}")
def test_quantifier_matches(test: TestRegexMatches):
    test.assert_matches('aaa')
    test.assert_doesnt_match('bbb', 'ccccc', 'a')


@TestRegexMatches(r"a{3}\Z", TestType.BOUNDARY)
def test_quantifier_matches(test: TestRegexMatches):
    test.assert_matches('aaa')
    test.assert_doesnt_match('aa', 'aaaa')


@TestRegexShape(r"a{3,}")
def test_min_quantifier(start: NodeMatcher):
    start.has_literal_chain('aaa', RegexState.END)\
        .has_literal('a', RegexState.SELF)


@TestRegexMatches(r"a{3,}")
def test_min_quantifier_matches(test: TestRegexMatches):
    test.assert_matches('aaa', 'aaaa', 'aaaaaaaa')
    test.assert_doesnt_match('bbb', 'ccccc', 'a')


@TestRegexMatches(r"a{3,}", TestType.BOUNDARY)
def test_min_quantifier_matches(test: TestRegexMatches):
    test.assert_matches('aaa', 'aaaa')
    test.assert_doesnt_match('aa')


@TestRegexShape(r"a{,3}")
def test_max_quantifier(start: NodeMatcher):
    start.has(MatchConditions.epsilon_transition, RegexState.END)
    first = start.has_literal('a')
    first.has(MatchConditions.epsilon_transition, RegexState.END)
    second = first.has_literal('a')
    second.has(MatchConditions.epsilon_transition, RegexState.END)
    second.has_literal('a', RegexState.END)


@TestRegexMatches(r"a{,3}\Z")
def test_max_quantifier_matches(test: TestRegexMatches):
    test.assert_matches('a', 'aa')
    test.assert_doesnt_match('bbb', 'aaaaaa')


@TestRegexMatches(r"a{,3}\Z", TestType.BOUNDARY)
def test_max_quantifier_matches(test: TestRegexMatches):
    test.assert_matches('aaa', 'aa')
    test.assert_doesnt_match('aaaa')


@TestRegexShape(r"a{3,5}")
def test_minmax_quantifier(start: NodeMatcher):
    first = start.has_literal_chain('aaa')  # first 3 compulsary
    first.has(MatchConditions.epsilon_transition, RegexState.END)
    second = first.has_literal('a')
    second.has(MatchConditions.epsilon_transition, RegexState.END)
    second.has_literal('a', RegexState.END)


@TestRegexMatches(r"a{3,5}\Z")
def test_minmax_quantifier_matches(test: TestRegexMatches):
    test.assert_matches('aaa', 'aaaa')
    test.assert_doesnt_match('bbb', 'aaaaaa', 'aa')


@TestRegexMatches(r"a{3,5}\Z", TestType.BOUNDARY)
def test_minmax_quantifier_matches(test: TestRegexMatches):
    test.assert_matches('aaa', 'aaaaa')
    test.assert_doesnt_match('aa', 'aaaaaa')


def run_tests():
    TestCase.run_cases()
    _copy_html(TestCase.produce_html_printout())
    print("Ran all tests: results in clipboard")
    TestRegexShape._failed_regex.display()
