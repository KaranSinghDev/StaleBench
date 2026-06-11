"""The default answer-checker must be robust: no substring false positives, phrase-aware."""
from stalebench.system import TokenChecker

c = TokenChecker()


def test_no_substring_false_positive():
    # "Park" must NOT be matched inside "Parkinson"
    assert c.classify("The director is Parkinson.", "Maria Lopez", "David Park") == "OTHER"


def test_phrase_match_new_and_old():
    assert c.classify("It is David Park now.", "Maria Lopez", "David Park") == "NEW"
    assert c.classify("Maria Lopez leads it.", "Maria Lopez", "David Park") == "OLD"


def test_both_and_other():
    assert c.classify("David Park replaced Maria Lopez.", "Maria Lopez", "David Park") == "BOTH"
    assert c.classify("I don't know.", "Maria Lopez", "David Park") == "OTHER"


def test_numbers_are_whole_tokens():
    assert c.classify("The price is 42.", "19", "42") == "NEW"
    assert c.classify("It costs 1900.", "19", "42") == "OTHER"   # 19 != token 1900


def test_word_order_insensitive_but_phrase_aware():
    # phrase containment, not bag-of-words: a stray "David" alone is not "David Park"
    assert c.classify("David is here.", "Maria Lopez", "David Park") == "OTHER"
