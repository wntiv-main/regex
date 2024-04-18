from src.regexutil import MatchConditions
from .regex_tests import NodeMatcher, RegexState, TestNoParseError, TestParseError, TestRegexMatches, TestRegexShape
from .test import TestCase, TestType, _copy_html


# Digits
@TestRegexShape(r"\d")
def test_digits(start: NodeMatcher):
    start.has_any('0123456789', RegexState.END)


TestRegexMatches(r"\d")                 \
    .assert_matches('0', '3', '7', '9') \
    .assert_doesnt_match('a', 'five', 'hello', '?')


# "Word" chars
@TestRegexShape(r"\w")
def test_digits(start: NodeMatcher):
    # all chars matched by \w
    start.has_any('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
                  '0123456789_',
                  RegexState.END)


TestRegexMatches(r"\w")                                \
    .assert_matches('0', '7', '_', 'a', 'E', 'j', 'X') \
    .assert_doesnt_match('$', '+', '&', '?', '-')


# Star
@TestRegexShape(r"a*")
def test_star(start: NodeMatcher):
    start.is_also(RegexState.END).has_literal('a', RegexState.SELF)


TestRegexMatches(r"a*\Z")                     \
    .assert_matches('', 'a', 'aaa', 'aaaaaa') \
    .assert_doesnt_match('njds', 'hello')

TestRegexMatches(r"a*", TestType.BOUNDARY) \
    .assert_matches('')

TestParseError(r'*abc')


@TestRegexShape(r"(a?b?c?)*",
                expecting="all epsilon-moves to be removed")
def test_epsilon_closure_and_minification(start: NodeMatcher):
    start.is_also(RegexState.END)          \
        .has_literal('a', RegexState.SELF) \
        .has_literal('b', RegexState.SELF) \
        .has_literal('c', RegexState.SELF)


# Plus
@TestRegexShape(r"a+")
def test_plus(start: NodeMatcher):
    end = start.has_literal('a', RegexState.END)
    end.has_literal('a', RegexState.SELF)


TestRegexMatches(r"a+")                                   \
    .assert_matches('a', 'aaa', 'aaaaaa', 'aaaaaabdsjmn') \
    .assert_doesnt_match('', 'njds', 'hello')

TestRegexMatches(r"a+", TestType.BOUNDARY) \
    .assert_matches('a')                   \
    .assert_doesnt_match('')

TestParseError(r'+abc')

# Optional
@TestRegexShape(r"a?")
def test_optional(start: NodeMatcher):
    start.has_literal('a', RegexState.END)
    start.has(MatchConditions.epsilon_transition, RegexState.END)


TestRegexMatches(r"a?\Z")    \
    .assert_matches('', 'a') \
    .assert_doesnt_match('aaa', 'aaaaaaa')

TestParseError(r'?abc')


# OR
@TestRegexShape(r"a|b")
def test_or(start: NodeMatcher):
    start.has_literal('a', RegexState.END)
    start.has_literal('b', RegexState.END)


TestRegexMatches(r"a|b")      \
    .assert_matches('a', 'b') \
    .assert_doesnt_match('c', 'd')


@TestRegexShape(r"hello|goodbye")
def test_longer_or(start: NodeMatcher):
    start.has_literal_chain('hello', RegexState.END)
    start.has_literal_chain('goodbye', RegexState.END)


@TestRegexShape(r"hello|hi",
                expecting="two h-moves to be merged into one")
def test_powerset_construction(start: NodeMatcher):
    next = start.has_literal('h')
    next.has_literal_chain('ello', RegexState.END)
    next.has_literal('i', RegexState.END)


@TestRegexShape(r"\da|[13579a-e]b",
                expecting="the edges to be merged into an equivilent"
                          "edge that covers both of them")
def test_complex_powerset_construction(start: NodeMatcher):
    left = start.has_any('02468')
    intersect = start.has_any('13579')
    right = start.has_any('abcde')
    left.has_literal('a', RegexState.END)
    intersect.has_literal('a', RegexState.END)
    intersect.has_literal('b', RegexState.END)
    right.has_literal('b', RegexState.END)


TestRegexMatches(r"hello\Z|goodbye\Z")  \
    .assert_matches('hello', 'goodbye') \
    .assert_doesnt_match('helloodbye', 'he', 'hellodbye')

TestParseError(r'|abc')
TestParseError(r'abc|')


# Char classes
@TestRegexShape(r"[ab]")
def test_class_specifier(start: NodeMatcher):
    start.has_any('ab', RegexState.END)


TestRegexMatches(r"[ab]")     \
    .assert_matches('a', 'b') \
    .assert_doesnt_match('c', 'd')

TestParseError(r'[')
TestParseError(r'[abc')
TestParseError(r'hello]')


@TestRegexShape(r"[^ab]")
def test_inverted_class_specifier(start: NodeMatcher):
    start.has_any_except('ab', RegexState.END)


TestRegexMatches(r"[^ab]")    \
    .assert_matches('c', 'd') \
    .assert_doesnt_match('a', 'b')


@TestRegexShape(r"[c-g]")
def test_class_specifier_range(start: NodeMatcher):
    start.has_any('cdefg', RegexState.END)


TestRegexMatches(r"[c-g]")         \
    .assert_matches('f', 'd', 'e') \
    .assert_doesnt_match('a', 'r', 'b', 'k', 'z')

TestRegexMatches(r"[c-g]", TestType.BOUNDARY) \
    .assert_matches('c', 'g')                 \
    .assert_doesnt_match('b', 'h')

TestParseError(r'[a-]')
TestParseError(r'[-a]')
TestNoParseError(r'[a\-]')
TestNoParseError(r'[\-a]')


# Quantifiers
TestParseError(r'a{hello}')
TestParseError(r'a{1,2,3}')


@TestRegexShape(r"a{3}")
def test_quantifier(start: NodeMatcher):
    start.has_literal_chain('aaa', RegexState.END)


TestRegexMatches(r"a{3}")  \
    .assert_matches('aaa') \
    .assert_doesnt_match('bbb', 'ccccc', 'a')

TestRegexMatches(r"a{3}\Z", TestType.BOUNDARY) \
    .assert_matches('aaa')                     \
    .assert_doesnt_match('aa', 'aaaa')

TestParseError(r'a{0}')
TestNoParseError(r'a{1}')


@TestRegexShape(r"a{3,}")
def test_min_quantifier(start: NodeMatcher):
    start.has_literal_chain('aaa', RegexState.END)\
        .has_literal('a', RegexState.SELF)


TestRegexMatches(r"a{3,}")                     \
    .assert_matches('aaa', 'aaaa', 'aaaaaaaa') \
    .assert_doesnt_match('bbb', 'ccccc', 'a')

TestRegexMatches(r"a{3,}", TestType.BOUNDARY) \
    .assert_matches('aaa', 'aaaa')            \
    .assert_doesnt_match('aa')

TestParseError(r'a{-1,}')
TestNoParseError(r'a{0,}')


@TestRegexShape(r"a{,3}")
def test_max_quantifier(start: NodeMatcher):
    start.has(MatchConditions.epsilon_transition, RegexState.END)
    first = start.has_literal('a')
    first.has(MatchConditions.epsilon_transition, RegexState.END)
    second = first.has_literal('a')
    second.has(MatchConditions.epsilon_transition, RegexState.END)
    second.has_literal('a', RegexState.END)


TestRegexMatches(r"a{,3}\Z")   \
    .assert_matches('a', 'aa') \
    .assert_doesnt_match('bbb', 'aaaaaa')

TestRegexMatches(r"a{,3}\Z", TestType.BOUNDARY) \
    .assert_matches('aaa', 'aa')                \
    .assert_doesnt_match('aaaa')

TestParseError(r'a{,0}')


@TestRegexShape(r"a{3,5}")
def test_minmax_quantifier(start: NodeMatcher):
    first = start.has_literal_chain('aaa')  # first 3 compulsary
    first.has(MatchConditions.epsilon_transition, RegexState.END)
    second = first.has_literal('a')
    second.has(MatchConditions.epsilon_transition, RegexState.END)
    second.has_literal('a', RegexState.END)


TestRegexMatches(r"a{3,5}\Z")      \
    .assert_matches('aaa', 'aaaa') \
    .assert_doesnt_match('bbb', 'aaaaaa', 'aa')

TestRegexMatches(r"a{3,5}\Z", TestType.BOUNDARY) \
    .assert_matches('aaa', 'aaaaa')              \
    .assert_doesnt_match('aa', 'aaaaaa')

TestParseError(r'a{5,3}')
TestNoParseError(r'a{3,5}')


def run_tests():
    print("TEST SUMMARY:")
    print(TestCase.format_results_table(TestCase.run_cases()))
    _copy_html(TestCase.produce_html_printout())
    print("Ran all tests: full results in clipboard")
    TestRegexShape._failed_regex.display()
