"""Tests for src.langid (SPEC §5.2)."""

from src.langid import COLUMNS, load_lexicons, tag_corpus, tag_sentence, tag_token


def test_load_lexicons():
    lexicons = load_lexicons()
    assert "hai" in lexicons["roman_urdu"]
    assert "meeting" in lexicons["english"]
    assert "میں" in lexicons["urdu"]
    assert "آئون" in lexicons["sindhi"]
    assert lexicons["ambiguous"] == {"do", "hi", "in", "is", "the", "to", "us"}


def test_pure_sindhi():
    tokens = ["آئون", "سڀاڻي", "يونيورسٽي", "ويندس", "."]
    assert tag_sentence(tokens) == ["SND", "SND", "SND", "SND", "OTHER"]


def test_pure_urdu():
    tokens = ["میں", "کل", "یونیورسٹی", "جاؤں", "گا", "۔"]
    assert tag_sentence(tokens) == ["URD", "URD", "URD", "URD", "URD", "OTHER"]


def test_pure_roman_urdu():
    tokens = ["mujhe", "kal", "ghar", "jana", "hai", "."]
    assert tag_sentence(tokens) == ["ROM", "ROM", "ROM", "ROM", "ROM", "OTHER"]


def test_pure_english():
    tokens = ["The", "meeting", "is", "very", "good", "."]
    assert tag_sentence(tokens) == ["ENG", "ENG", "ENG", "ENG", "ENG", "OTHER"]


def test_mixed_sentence():
    tokens = [
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
    assert tag_sentence(tokens) == [
        "ROM",
        "ENG",
        "ROM",
        "ROM",
        "OTHER",
        "URD",
        "URD",
        "URD",
        "URD",
        "OTHER",
    ]


def test_ambiguous_uses_left_neighbour():
    assert tag_sentence(["mujhe", "do", "kal"]) == ["ROM", "ROM", "ROM"]


def test_ambiguous_uses_right_neighbour_when_no_left():
    assert tag_sentence(["do", "meeting"]) == ["ENG", "ENG"]


def test_ambiguous_skips_other_when_finding_neighbour():
    assert tag_sentence(["do", ".", "kal"]) == ["ROM", "OTHER", "ROM"]


def test_ambiguous_without_context_falls_back_to_english():
    assert tag_sentence(["do"]) == ["ENG"]
    assert tag_sentence(["do", "hi"]) == ["ENG", "ENG"]
    assert tag_token("do") == "ENG"


def test_tag_token_uses_prev_tag_for_ambiguous_word():
    assert tag_token("do", prev_tag="ROM") == "ROM"
    assert tag_token("do", prev_tag="OTHER") == "ENG"


def test_emoji_url_mention_hashtag_number_are_other():
    tokens = ["Check", "https://x.com", "@ravi", "#tag", "42", "😊"]
    assert tag_sentence(tokens) == [
        "ENG",
        "OTHER",
        "OTHER",
        "OTHER",
        "OTHER",
        "OTHER",
    ]


def test_tag_corpus_dataframe_schema_and_session_values():
    df = tag_corpus("Mera laptop kharab hai.\nمیں کل جاؤں گا۔")
    assert list(df.columns) == COLUMNS
    assert set(df["sent_id"]) == {0, 1}
    assert df[df["token"] == "Mera"]["norm"].item() == "mera"
    assert (df["manual_lang"] == "").all()
    assert (df["final_lang"] == df["auto_lang"]).all()


def test_tag_corpus_empty_input():
    df = tag_corpus("")
    assert df.empty
    assert list(df.columns) == COLUMNS
