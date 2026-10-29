"""CodeMix Corpus Lab — Streamlit entrypoint."""

import html
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.langid import tag_corpus
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


def render_input() -> None:
    st.title("Input")
    st.caption("Paste text, upload a .txt/.csv file, or load the sample corpus.")

    if st.button("Load sample corpus"):
        if SAMPLE_PATH.is_file():
            st.session_state["raw_text"] = SAMPLE_PATH.read_text(encoding="utf-8")
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

    df = build_token_df(text)
    st.session_state["tokens_df"] = df

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
    st.info("Placeholder: Frequency, Concordance, N-grams, and Collocations.")


def render_correct_evaluate() -> None:
    st.title("Correct & Evaluate")
    st.info("Placeholder: editable tag table and accuracy report.")


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
