# SPEC: CodeMix Corpus Lab

Technical specification. Read `PRD.md` first for goals and scope.

---

## 1. Overview

A Streamlit app for analysing code-mixed Sindhi / Urdu / Roman Urdu / English text. Pipeline:

```
Input -> Tokenize -> Language tag -> Switch analysis -> Corpus tools -> Visualise -> Correct -> Export
```

## 2. Tech Stack

- Python 3.11+
- streamlit, pandas, plotly, nltk, pytest
- No database; state lives in `st.session_state` and exported files

## 3. Project Structure

```
codemix-lab/
├── app.py
├── requirements.txt
├── README.md
├── PRD.md
├── SPEC.md
├── src/
│   ├── __init__.py
│   ├── tokenize.py
│   ├── langid.py
│   ├── switching.py
│   ├── corpus_tools.py
│   ├── evaluate.py
│   └── export.py
├── data/
│   ├── lexicons/
│   │   ├── english.txt
│   │   ├── roman_urdu.txt
│   │   ├── urdu.txt
│   │   └── sindhi.txt
│   └── sample/
│       └── sample_corpus.txt
└── tests/
    ├── test_tokenize.py
    ├── test_langid.py
    ├── test_switching.py
    └── test_corpus_tools.py
```

## 4. Data Model

Core object is a token-level DataFrame:

| Column | Type | Description |
|---|---|---|
| sent_id | int | Sentence index (starting at 0) |
| tok_id | int | Token index within sentence |
| token | str | Original token |
| norm | str | Normalised form (lowercase, repeated-letter collapse for Roman Urdu) |
| auto_lang | str | One of SND, URD, ROM, ENG, OTHER |
| manual_lang | str | Corrected tag (empty until edited) |
| final_lang | str | manual_lang if set, else auto_lang |

## 5. Modules

### 5.1 `tokenize.py`
- `split_sentences(text) -> list[str]`: split on newline (one sentence per line); optional split on `.`, `?`, `!`, `۔`, `؟`
- `tokenize(sentence) -> list[str]`: regex split that keeps Arabic-script words intact, separates punctuation, keeps hashtags/mentions/URLs as OTHER
- Normalise Unicode (NFC) and strip zero-width characters where appropriate

### 5.2 `langid.py`
Tagging rules, applied in order:

1. Numbers, punctuation, emoji, URLs, mentions -> `OTHER`
2. **Arabic-script token:**
   - Contains Sindhi-specific letters (ڄ ڃ ڇ ڏ ڊ ڍ ڌ ڙ ڻ ٺ ٽ ٿ ڀ ڦ ڪ ڳ ڱ ھ) -> `SND`
   - Else in `sindhi.txt` -> `SND`
   - Else -> `URD`
3. **Latin-script token** (after normalisation):
   - In `roman_urdu.txt` -> `ROM`
   - In `english.txt` -> `ENG`
   - In both -> resolve by context (neighbour tags); fall back to `ENG`
   - In neither -> use character n-gram heuristic; fall back to `OTHER`

API:
- `tag_token(token, prev_tag=None) -> str`
- `tag_sentence(tokens) -> list[str]`
- `load_lexicons(path) -> dict[str, set[str]]`

### 5.3 `switching.py`
- `find_switches(tags) -> list[int]`: positions where tag changes, ignoring `OTHER`
- `classify_sentence(tags) -> "monolingual" | "intra" | "single-lang"`
- `inter_sentential(df) -> list`: sentences where the dominant language differs from the previous sentence
- `switch_rate(df) -> float`: switches per 100 tokens
- `summary(df) -> dict`: language distribution, switch rate, switch-type counts, switch-direction matrix (e.g. ENG->ROM)

### 5.4 `corpus_tools.py`
- `frequency(df, lang=None) -> DataFrame`
- `concordance(df, keyword, window=5) -> DataFrame` (KWIC: left, keyword, right, sent_id, lang)
- `ngrams(df, n=2, lang=None) -> DataFrame`
- `collocations(df, keyword, window=4, measure="pmi", min_freq=2) -> DataFrame`

### 5.5 `evaluate.py`
- `accuracy_report(df) -> dict`: compares `auto_lang` against `manual_lang` on edited rows (or a fully annotated gold file); returns accuracy, per-language precision/recall/F1, confusion matrix

### 5.6 `export.py`
- `to_csv(df) -> bytes`
- `to_json(df) -> bytes`: nested by sentence, with tags and summary stats

## 6. UI (Streamlit)

Sidebar navigation with 6 pages:

1. **Input:** text area, file upload, "Load sample corpus" button
2. **Tagging:** colour-coded tokens per sentence (SND, URD, ROM, ENG, OTHER) plus language distribution bar chart
3. **Switching:** summary metrics (switch rate, types), per-sentence switch visualisation, direction matrix heatmap
4. **Corpus Tools:** tabs for Frequency, Concordance, N-grams, Collocations with filters by language
5. **Correct & Evaluate:** `st.data_editor` for tag corrections, then accuracy report and confusion matrix
6. **Export:** download annotated CSV and JSON

Rules: RTL text rendered with `direction: rtl` styling; results cached with `st.cache_data`; clear error messages for empty or malformed input.

## 7. Lexicons

- Plain text, one word per line, UTF-8, lowercase
- `roman_urdu.txt`: start with about 1,000 high-frequency words and grow from annotated data
- `sindhi.txt` / `urdu.txt`: common function words and frequent vocabulary
- `english.txt`: common English wordlist (e.g. NLTK words or a frequency list)
- Keep a `CONTRIBUTING` note on how to add words

## 8. Testing

- Unit tests for each module with pytest
- Minimum cases: pure Sindhi, pure Urdu, pure Roman Urdu, pure English, mixed sentence, emoji/URL/number, empty input
- Gold set: 200-500 manually annotated sentences used for the accuracy report

## 9. Deployment

- `requirements.txt` pinned
- Deploy to Streamlit Community Cloud from GitHub
- README includes: overview, screenshots, install, usage, data note, research note, license

## 10. Acceptance Criteria

- [ ] `streamlit run app.py` works from a clean clone
- [ ] Sample corpus loads and tags with no errors
- [ ] All 6 pages functional
- [ ] Token-level accuracy of 85% or higher on the gold set
- [ ] Export files open correctly in Excel/pandas
- [ ] All tests pass
- [ ] Live demo URL in README

## 11. Future (V2)

- Roman Sindhi support, custom lexicon upload, simple ML tagger trained on the gold set, Praat-style audio module, PDF report export
