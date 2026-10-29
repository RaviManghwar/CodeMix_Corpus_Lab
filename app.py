"""CodeMix Corpus Lab — Streamlit entrypoint."""

import html
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.corpus_tools import collocations, concordance, frequency, ngrams
from src.evaluate import accuracy_report, list_errors
from src.langid import LANGS, OTHER, load_lexicons, tag_corpus
from src.switching import classify_sentence, find_switches, summary

SAMPLE_PATH = Path(__file__).parent / "data" / "sample" / "sample_corpus.txt"

PAGES = [
    "Input",
    "Tagging",
    "Switching",
    "Corpus Tools",
    "Correct & Evaluate",
    "Export",
]

LANG_COLORS = {
    "SND": "#FFD6A5",
    "URD": "#B7E4C7",
    "ROM": "#FDFFB6",
    "ENG": "#A2D2FF",
    "OTHER": "#DEE2E6",
}
LANG_LABELS = {
    "SND": "Sindhi",
    "URD": "Urdu",
    "ROM": "Roman Urdu",
    "ENG": "English",
    "OTHER": "Other",
}
_ARABIC_SCRIPT = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]"
)


@st.cache_data(show_spinner=False)
def build_token_df(text: str) -> pd.DataFrame:
    return tag_corpus(text)


def _is_rtl(token: str) -> bool:
    return bool(_ARABIC_SCRIPT.search(token))


LANGUAGE_FILTERS = ["All", "SND", "URD", "ROM", "ENG"]


def _language_filter(key: str) -> str | None:
    selection = st.selectbox(
        "Language filter",
        LANGUAGE_FILTERS,
        format_func=lambda value: (
            "All" if value == "All" else f"{value} ({LANG_LABELS[value]})"
        ),
        key=key,
    )
    return None if selection == "All" else selection


def _keyword_found(df: pd.DataFrame, keyword: str, lang: str | None = None) -> bool:
    data = df[df["final_lang"] != OTHER]
    if lang is not None:
        data = data[data["final_lang"] == lang]
    return bool(
        (data["norm"].str.casefold() == keyword.strip().casefold()).any()
    )


def _download_csv(results: pd.DataFrame, filename: str, key: str) -> None:
    st.download_button(
        "Download CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        disabled=results.empty,
        key=key,
    )


def render_input() -> None:
    st.title("Input")
    st.caption("Paste text, upload a .txt/.csv file, or load the sample corpus.")

    if st.button("Load sample corpus"):
        if SAMPLE_PATH.is_file():
            st.session_state["raw_text"] = SAMPLE_PATH.read_text(encoding="utf-8")
            st.session_state.pop("tokens_df", None)
            st.session_state.pop("tagged_text", None)
        else:
            st.error(f"Sample corpus not found at {SAMPLE_PATH}")

    if st.session_state.get("raw_text"):
        st.text_area("Corpus", st.session_state["raw_text"], height=300)


def render_tagging() -> None:
    st.title("Tagging")

    text = st.session_state.get("raw_text", "")
    if not text.strip():
        st.info("No corpus loaded yet. Go to Input to paste text or load the sample.")
        return

    df = st.session_state.get("tokens_df")
    if df is None or st.session_state.get("tagged_text") != text:
        df = build_token_df(text)
        st.session_state["tokens_df"] = df
        st.session_state["tagged_text"] = text

    legend = "".join(
        f'<span style="background:{color};color:#212529;padding:2px 8px;'
        'border-radius:6px;margin-right:6px;font-size:0.85rem;">'
        f"{code} — {LANG_LABELS[code]}</span>"
        for code, color in LANG_COLORS.items()
    )
    st.markdown(f'<div style="margin-bottom:0.75rem;">{legend}</div>', unsafe_allow_html=True)

    for _, group in df.groupby("sent_id", sort=True):
        spans = []
        for row in group.itertuples(index=False):
            rtl = ";direction:rtl;unicode-bidi:isolate" if _is_rtl(row.token) else ""
            color = LANG_COLORS.get(row.final_lang, LANG_COLORS["OTHER"])
            spans.append(
                f'<span title="{row.final_lang}" style="background:{color};'
                "color:#212529;padding:1px 5px;border-radius:4px;margin:0 2px;"
                f'display:inline-block{rtl};">{html.escape(row.token)}</span>'
            )
        st.markdown(
            f'<div style="line-height:2;margin-bottom:0.5rem;">{"".join(spans)}</div>',
            unsafe_allow_html=True,
        )

    st.subheader("Language distribution")
    counts = df["final_lang"].value_counts().reindex(LANG_COLORS, fill_value=0)
    chart_df = counts.rename_axis("Language").reset_index(name="Tokens")
    fig = px.bar(
        chart_df,
        x="Language",
        y="Tokens",
        color="Language",
        color_discrete_map=LANG_COLORS,
        text="Tokens",
    )
    fig.update_layout(
        showlegend=False,
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, width="stretch")


def render_switching() -> None:
    st.title("Switching")

    df = st.session_state.get("tokens_df")
    if df is None or df.empty:
        st.info(
            "No tagged corpus yet. Load a corpus on the Input page and open "
            "Tagging before viewing switching statistics."
        )
        return

    stats = summary(df)
    sentence_rows = []
    for sent_id, group in df.groupby("sent_id", sort=True):
        tags = group["final_lang"].tolist()
        sentence_rows.append(
            {
                "Sentence": int(sent_id) + 1,
                "Tokens": " ".join(group["token"]),
                "Switches": len(find_switches(tags)),
                "Type": classify_sentence(tags),
            }
        )
    sentences_df = pd.DataFrame(sentence_rows)
    mixed_pct = (
        (sentences_df["Type"] == "intra").mean() * 100
        if not sentences_df.empty
        else 0.0
    )

    rate_col, mixed_col, total_col = st.columns(3)
    rate_col.metric("Switch rate", f"{stats['switch_rate']:.1f} / 100 tokens")
    mixed_col.metric("Mixed sentences", f"{mixed_pct:.0f}%")
    total_col.metric(
        "Total switches",
        stats["intra_sentential"] + stats["inter_sentential"],
    )
    st.caption(
        f"{stats['intra_sentential']} intra-sentential · "
        f"{stats['inter_sentential']} inter-sentential"
    )

    st.subheader("Language distribution")
    distribution = stats["distribution"]
    bar_df = pd.DataFrame(
        {"Language": list(distribution), "Tokens": list(distribution.values())}
    )
    bar = px.bar(
        bar_df,
        x="Language",
        y="Tokens",
        color="Language",
        color_discrete_map=LANG_COLORS,
        text="Tokens",
    )
    bar.update_layout(
        showlegend=False,
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(bar, width="stretch")

    st.subheader("Switch directions")
    languages = ["SND", "URD", "ROM", "ENG"]
    matrix = stats["direction_matrix"]
    matrix_df = pd.DataFrame(
        [[matrix[source][target] for target in languages] for source in languages],
        index=languages,
        columns=languages,
    )
    heatmap = px.imshow(
        matrix_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        labels={"x": "To", "y": "From", "color": "Switches"},
    )
    heatmap.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(heatmap, width="stretch")

    st.subheader("Sentences")
    st.dataframe(sentences_df, width="stretch", hide_index=True)


def render_corpus_tools() -> None:
    st.title("Corpus Tools")

    df = st.session_state.get("tokens_df")
    if df is None or df.empty:
        st.info(
            "No tagged corpus yet. Load a corpus on the Input page and open "
            "Tagging before using the corpus tools."
        )
        return

    frequency_tab, concordance_tab, ngram_tab, collocation_tab = st.tabs(
        ["Frequency", "Concordance", "N-grams", "Collocations"]
    )

    with frequency_tab:
        lang = _language_filter("frequency_lang")
        results = frequency(df, lang=lang)
        if results.empty:
            st.info("No tokens match this language filter.")
        else:
            st.dataframe(
                results,
                width="stretch",
                hide_index=True,
                column_config={
                    "relative_freq_per_1000": st.column_config.NumberColumn(
                        "Per 1,000", format="%.2f"
                    )
                },
            )
        _download_csv(results, "frequency.csv", "frequency_csv")

    with concordance_tab:
        keyword = st.text_input("Keyword", key="concordance_keyword")
        controls = st.columns(2)
        window = controls[0].number_input(
            "Window size",
            min_value=1,
            max_value=50,
            value=5,
            key="concordance_window",
        )
        with controls[1]:
            lang = _language_filter("concordance_lang")

        results = concordance(df, keyword, window=int(window), lang=lang)
        if not keyword.strip():
            st.info("Enter a keyword to build a concordance.")
        elif results.empty:
            if _keyword_found(df, keyword, lang):
                st.info("No matches for this keyword in the selected language.")
            else:
                st.info(f"Keyword '{keyword.strip()}' was not found in the corpus.")
        else:
            st.dataframe(results, width="stretch", hide_index=True)
        _download_csv(results, "concordance.csv", "concordance_csv")

    with ngram_tab:
        lang = _language_filter("ngram_lang")
        controls = st.columns(2)
        n = controls[0].number_input(
            "N-gram size",
            min_value=1,
            max_value=5,
            value=2,
            key="ngram_size",
        )
        min_freq = controls[1].number_input(
            "Minimum frequency",
            min_value=1,
            value=1,
            key="ngram_min_freq",
        )

        results = ngrams(
            df,
            n=int(n),
            lang=lang,
            min_freq=int(min_freq),
        )
        if results.empty:
            st.info("No n-grams found with the current settings.")
        else:
            st.dataframe(results, width="stretch", hide_index=True)
        _download_csv(results, "ngrams.csv", "ngram_csv")

    with collocation_tab:
        keyword = st.text_input("Keyword", key="collocation_keyword")
        controls = st.columns(3)
        window = controls[0].number_input(
            "Window size",
            min_value=1,
            max_value=50,
            value=4,
            key="collocation_window",
        )
        min_freq = controls[1].number_input(
            "Minimum frequency",
            min_value=1,
            value=2,
            key="collocation_min_freq",
        )
        with controls[2]:
            lang = _language_filter("collocation_lang")

        scoped = df if lang is None else df[df["final_lang"] == lang]
        results = collocations(
            scoped,
            keyword,
            window=int(window),
            min_freq=int(min_freq),
        )
        if not keyword.strip():
            st.info("Enter a keyword to find collocates.")
        elif results.empty:
            if _keyword_found(scoped, keyword):
                st.info("No collocates meet the current settings.")
            else:
                st.info(f"Keyword '{keyword.strip()}' was not found in the corpus.")
        else:
            st.dataframe(
                results,
                width="stretch",
                hide_index=True,
                column_config={
                    "pmi": st.column_config.NumberColumn("PMI", format="%.2f")
                },
            )
        _download_csv(results, "collocations.csv", "collocation_csv")


def render_correct_evaluate() -> None:
    st.title("Correct & Evaluate")

    df = st.session_state.get("tokens_df")
    if df is None or df.empty:
        st.info(
            "No tagged corpus yet. Load a corpus on the Input page and open "
            "Tagging before correcting or evaluating tags."
        )
        return

    ambiguous = load_lexicons()["ambiguous"]
    sent_ids = sorted(int(sent_id) for sent_id in df["sent_id"].unique())
    sentence_options = ["All sentences"] + [
        f"Sentence {sent_id + 1}" for sent_id in sent_ids
    ]
    controls = st.columns(2)
    with controls[0]:
        selected_sentence = st.selectbox(
            "Sentence filter",
            sentence_options,
            key="evaluate_sentence_filter",
        )
    with controls[1]:
        only_other = st.checkbox(
            "Show only OTHER / ambiguous tokens",
            key="evaluate_only_other",
        )

    display = df
    if selected_sentence != "All sentences":
        target = sent_ids[sentence_options.index(selected_sentence) - 1]
        display = display[display["sent_id"] == target]
    if only_other:
        display = display[
            (display["auto_lang"] == OTHER)
            | (display["norm"].str.casefold().isin(ambiguous))
        ]

    if display.empty:
        st.info("No tokens match the current filters.")
    else:
        editor_df = display[
            ["sent_id", "tok_id", "token", "auto_lang", "manual_lang"]
        ].copy()
        editor_df["manual_lang"] = (
            editor_df["manual_lang"].fillna("").astype(str)
        )
        edited = st.data_editor(
            editor_df,
            key=f"annotation_editor_{selected_sentence}_{only_other}",
            hide_index=True,
            width="stretch",
            disabled=["sent_id", "tok_id", "token", "auto_lang"],
            column_config={
                "manual_lang": st.column_config.SelectboxColumn(
                    "Manual language",
                    options=[*LANGS, ""],
                    help="Leave blank to use the automatic tag.",
                )
            },
        )
        updates = edited[["sent_id", "tok_id", "manual_lang"]].copy()
        updates["manual_lang"] = (
            updates["manual_lang"].fillna("").astype(str)
        )
        updates = updates.rename(columns={"manual_lang": "manual_update"})
        updated = df.merge(updates, on=["sent_id", "tok_id"], how="left")
        updated["manual_lang"] = (
            updated["manual_update"]
            .fillna(updated["manual_lang"])
            .fillna("")
            .astype(str)
            .str.strip()
        )
        updated = updated.drop(columns="manual_update")
        updated["final_lang"] = updated["manual_lang"].where(
            updated["manual_lang"].isin(LANGS), updated["auto_lang"]
        )
        st.session_state["tokens_df"] = updated
        df = updated

    gold_col = "manual_lang"
    eval_df = df
    source = "manual corrections"
    uploaded = st.file_uploader(
        "Load gold file (CSV with sent_id, tok_id, gold_lang)",
        type=["csv"],
        key="gold_file",
    )
    if uploaded is not None:
        try:
            gold = pd.read_csv(uploaded)
        except Exception:
            st.error("Could not read the gold file as CSV.")
        else:
            required = {"sent_id", "tok_id", "gold_lang"}
            if not required.issubset(gold.columns):
                st.error(
                    "Gold file must contain sent_id, tok_id, and gold_lang columns."
                )
            else:
                gold = gold[["sent_id", "tok_id", "gold_lang"]].copy()
                gold["sent_id"] = pd.to_numeric(gold["sent_id"], errors="coerce")
                gold["tok_id"] = pd.to_numeric(gold["tok_id"], errors="coerce")
                gold = gold.dropna(subset=["sent_id", "tok_id"])
                gold["sent_id"] = gold["sent_id"].astype(int)
                gold["tok_id"] = gold["tok_id"].astype(int)
                gold = gold.drop_duplicates(
                    subset=["sent_id", "tok_id"], keep="last"
                )
                eval_df = df.merge(
                    gold,
                    on=["sent_id", "tok_id"],
                    how="left",
                    sort=False,
                )
                gold_col = "gold_lang"
                source = "the uploaded gold file"
                matched = int(
                    eval_df["gold_lang"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .ne("")
                    .sum()
                )
                st.caption(
                    f"Loaded {len(gold)} gold rows; {matched} matched the token table."
                )

    report = accuracy_report(eval_df, gold_col=gold_col)
    errors = list_errors(eval_df, gold_col=gold_col)

    st.caption(f"Comparing automatic tags against {source}.")
    accuracy_col, annotated_col = st.columns(2)
    accuracy_col.metric("Accuracy", f"{report['accuracy'] * 100:.1f}%")
    annotated_col.metric(
        "Annotated",
        f"{report['annotated_tokens']} / {report['total_tokens']} "
        f"({report['annotated_pct']:.0f}%)",
    )
    if report["annotated_tokens"] < 100:
        st.warning(
            f"Only {report['annotated_tokens']} tokens are annotated — "
            "accuracy is not reliable yet (100+ recommended)."
        )

    st.subheader("Per-language metrics")
    st.dataframe(
        report["per_language"],
        width="stretch",
        hide_index=True,
        column_config={
            "precision": st.column_config.NumberColumn(format="%.3f"),
            "recall": st.column_config.NumberColumn(format="%.3f"),
            "f1": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    st.subheader("Confusion matrix")
    confusion = report["confusion_matrix"]
    figure = px.imshow(
        confusion,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        labels={"x": "Predicted (auto)", "y": "Gold", "color": "Tokens"},
    )
    figure.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(figure, width="stretch")

    st.subheader("Errors")
    st.dataframe(errors, width="stretch", hide_index=True)
    _download_csv(errors, "errors.csv", "errors_csv")


def render_export() -> None:
    st.title("Export")
    st.info("Placeholder: download annotated CSV and JSON.")


RENDERERS = {
    "Input": render_input,
    "Tagging": render_tagging,
    "Switching": render_switching,
    "Corpus Tools": render_corpus_tools,
    "Correct & Evaluate": render_correct_evaluate,
    "Export": render_export,
}


def main() -> None:
    st.set_page_config(page_title="CodeMix Corpus Lab", layout="wide")
    st.sidebar.title("CodeMix Corpus Lab")
    page = st.sidebar.radio("Pages", PAGES)
    RENDERERS[page]()


if __name__ == "__main__":
    main()
