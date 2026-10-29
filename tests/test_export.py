"""Tests for src.export (SPEC §5.6)."""

import json
from io import BytesIO

import pandas as pd

from src.export import EXPORT_COLUMNS, to_csv, to_json
from src.langid import COLUMNS


def make_df(sentences: list[list[tuple[str, str]]]) -> pd.DataFrame:
    rows = []
    for sent_id, tokens in enumerate(sentences):
        for tok_id, (token, tag) in enumerate(tokens):
            rows.append(
                {
                    "sent_id": sent_id,
                    "tok_id": tok_id,
                    "token": token,
                    "norm": token.lower(),
                    "auto_lang": tag,
                    "manual_lang": "",
                    "final_lang": tag,
                }
            )
    return pd.DataFrame(rows, columns=COLUMNS)


def test_csv_round_trip_with_bom_and_unicode():
    df = make_df(
        [
            [("Kal", "ROM"), ("meeting", "ENG"), ("hai", "ROM")],
            [("میں", "URD"), ("کل", "URD")],
        ]
    )
    df.loc[0, "manual_lang"] = "ROM"

    data = to_csv(df)

    assert data.startswith(b"\xef\xbb\xbf")
    result = pd.read_csv(
        BytesIO(data), encoding="utf-8-sig", keep_default_na=False
    )
    assert list(result.columns) == EXPORT_COLUMNS
    assert result["token"].tolist() == ["Kal", "meeting", "hai", "میں", "کل"]
    assert result["manual_lang"].tolist() == ["ROM", "", "", "", ""]
    assert result["final_lang"].tolist() == df["final_lang"].tolist()


def test_json_structure_summary_and_unicode():
    df = make_df(
        [
            [("Kal", "ROM"), ("meeting", "ENG"), ("hai", "ROM")],
            [("میں", "URD"), ("کل", "URD")],
        ]
    )
    raw = to_json(df)
    payload = json.loads(raw.decode("utf-8"))

    assert "میں" in raw.decode("utf-8")
    assert set(payload) == {"summary", "sentences"}
    assert payload["summary"]["distribution"] == {
        "SND": 0,
        "URD": 2,
        "ROM": 2,
        "ENG": 1,
        "OTHER": 0,
    }
    assert payload["summary"]["intra_sentential"] == 2

    first, second = payload["sentences"]
    assert first["sent_id"] == 0
    assert first["text"] == "Kal meeting hai"
    assert first["switch_count"] == 2
    assert [token["token"] for token in first["tokens"]] == [
        "Kal",
        "meeting",
        "hai",
    ]
    assert first["tokens"][1]["auto_lang"] == "ENG"
    assert first["tokens"][1]["manual_lang"] == ""

    assert second["sent_id"] == 1
    assert second["text"] == "میں کل"
    assert second["switch_count"] == 0
    assert [token["token"] for token in second["tokens"]] == ["میں", "کل"]


def test_json_uses_original_text_when_provided():
    text = "Kal meeting hai.\nمیں کل گیا۔"
    df = make_df(
        [
            [("Kal", "ROM"), ("meeting", "ENG"), ("hai", "ROM"), (".", "OTHER")],
            [("میں", "URD"), ("کل", "URD"), ("گیا", "URD"), ("۔", "OTHER")],
        ]
    )
    payload = json.loads(to_json(df, text=text).decode("utf-8"))

    assert payload["sentences"][0]["text"] == "Kal meeting hai."
    assert payload["sentences"][1]["text"] == "میں کل گیا۔"


def test_json_switch_count_uses_final_lang():
    df = make_df(
        [
            [("Kal", "ROM"), ("meeting", "ENG"), ("hai", "ROM")],
        ]
    )
    df.loc[1, "manual_lang"] = "ROM"
    df.loc[1, "final_lang"] = "ROM"

    payload = json.loads(to_json(df).decode("utf-8"))

    assert payload["sentences"][0]["switch_count"] == 0
    assert payload["summary"]["intra_sentential"] == 0
    assert payload["sentences"][0]["tokens"][1]["manual_lang"] == "ROM"
    assert payload["sentences"][0]["tokens"][1]["final_lang"] == "ROM"


def test_empty_dataframe_exports():
    empty = pd.DataFrame(columns=COLUMNS)

    csv_bytes = to_csv(empty)
    result = pd.read_csv(BytesIO(csv_bytes), encoding="utf-8-sig")
    assert list(result.columns) == EXPORT_COLUMNS
    assert result.empty

    payload = json.loads(to_json(empty).decode("utf-8"))
    assert payload["sentences"] == []
    assert payload["summary"]["intra_sentential"] == 0
    assert payload["summary"]["switch_rate"] == 0.0
    assert all(
        count == 0 for count in payload["summary"]["distribution"].values()
    )
