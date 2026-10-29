"""Tests for src.tokenize (SPEC §5.1)."""

from src.tokenize import normalize, split_sentences, tokenize


def test_split_sentences_on_newlines():
    text = "Kal meeting hai.\nMujhe assignment submit karni hai.\n\nAaj chai piyein?"
    assert split_sentences(text) == [
        "Kal meeting hai.",
        "Mujhe assignment submit karni hai.",
        "Aaj chai piyein?",
    ]


def test_split_sentences_optional_terminators():
    text = "Hello. How are you? Fine! میں جاؤں گا۔ آپ کیسے ہیں؟"
    assert split_sentences(text, split_on_terminators=True) == [
        "Hello.",
        "How are you?",
        "Fine!",
        "میں جاؤں گا۔",
        "آپ کیسے ہیں؟",
    ]


def test_split_sentences_empty_input():
    assert split_sentences("") == []
    assert split_sentences("  \n\n\t ") == []


def test_tokenize_empty_input():
    assert tokenize("") == []
    assert tokenize("   ") == []


def test_tokenize_pure_english():
    assert tokenize("The quick brown fox.") == [
        "The",
        "quick",
        "brown",
        "fox",
        ".",
    ]


def test_tokenize_pure_roman_urdu():
    assert tokenize("Mujhe assignment submit karni hai.") == [
        "Mujhe",
        "assignment",
        "submit",
        "karni",
        "hai",
        ".",
    ]


def test_tokenize_pure_urdu():
    assert tokenize("میں کل یونیورسٹی جاؤں گا۔") == [
        "میں",
        "کل",
        "یونیورسٹی",
        "جاؤں",
        "گا",
        "۔",
    ]


def test_tokenize_pure_sindhi():
    assert tokenize("آئون سڀاڻي يونيورسٽي ويندس.") == [
        "آئون",
        "سڀاڻي",
        "يونيورسٽي",
        "ويندس",
        ".",
    ]


def test_tokenize_mixed_sentence():
    assert tokenize("Mera laptop kharab hai, میں کل جاؤں گا۔") == [
        "Mera",
        "laptop",
        "kharab",
        "hai",
        ",",
        "میں",
        "کل",
        "جاؤں",
        "گا",
        "۔",
    ]


def test_tokenize_punctuation_separated():
    assert tokenize("Hello, world! (test)") == [
        "Hello",
        ",",
        "world",
        "!",
        "(",
        "test",
        ")",
    ]


def test_tokenize_keeps_urls_mentions_hashtags_together():
    assert tokenize("Check https://example.com/x?y=1 @ravi #CodeMix") == [
        "Check",
        "https://example.com/x?y=1",
        "@ravi",
        "#CodeMix",
    ]


def test_tokenize_numbers_and_emoji():
    assert tokenize("I have 2 apples 😊 and 3.5 rotis!") == [
        "I",
        "have",
        "2",
        "apples",
        "😊",
        "and",
        "3.5",
        "rotis",
        "!",
    ]


def test_normalize_applies_nfc():
    assert normalize("cafe\u0301") == "café"
    assert tokenize("cafe\u0301") == ["café"]


def test_normalize_strips_zero_width_characters():
    assert normalize("hel\u200blo\ufeff") == "hello"
    assert tokenize("میں\u200b کل") == ["میں", "کل"]
