# CodeMix Corpus Lab

A browser-based lab for tagging and analysing code-mixed Sindhi / Urdu / Roman Urdu / English text.

Live demo: `<add Streamlit link>`

## Why this tool

Code-switching between Sindhi, Urdu, Roman Urdu and English is common in Pakistani digital communication, but standard corpus tools have no word-level language tags. Sindhi and Roman Urdu are also low-resource, so researchers often annotate mixed text manually in spreadsheets. CodeMix Corpus Lab puts tokenisation, language tagging, switch statistics, corpus tools, correction and export in one reproducible Streamlit app.

## Features

- Paste text or upload `.txt`/`.csv` files (one sentence per line), or load the built-in sample corpus.
- Unicode-safe tokenisation with NFC normalisation and zero-width character cleanup.
- Word-level tags: `SND`, `URD`, `ROM`, `ENG`, `OTHER`.
- Code-switching analysis: switch rate, intra-/inter-sentential switches, and a direction matrix.
- Corpus tools: frequency lists, KWIC concordance, N-grams, and PMI collocations with language filters.
- Editable annotation table with a gold-file option, accuracy, per-language precision/recall/F1, and a confusion matrix.
- Annotated CSV and JSON export; the CSV is UTF-8 with BOM for Excel.
- Right-to-left rendering for Sindhi and Urdu tokens.

## Screenshots

_Placeholder: add screenshots of the Input, Tagging, Switching, Corpus Tools, Correct & Evaluate, and Export pages here._

## Install and run

Requires Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Run the tests with `python3 -m pytest`.

## Project structure

```
.
├── app.py
├── requirements.txt
├── PRD (1).md
├── SPEC (2).md
├── data/
│   ├── lexicons/
│   │   ├── english.txt
│   │   ├── roman_urdu.txt
│   │   ├── urdu.txt
│   │   ├── sindhi.txt
│   │   └── ambiguous.txt
│   └── sample/
│       └── sample_corpus.txt
├── src/
│   ├── tokenize.py
│   ├── langid.py
│   ├── switching.py
│   ├── corpus_tools.py
│   ├── evaluate.py
│   └── export.py
└── tests/
    ├── test_tokenize.py
    ├── test_langid.py
    ├── test_switching.py
    ├── test_corpus_tools.py
    ├── test_evaluate.py
    └── test_export.py
```

## How tagging works

Tags are assigned in this order:

1. Numbers, punctuation, emoji, URLs, `@mentions`, and `#hashtags` become `OTHER`.
2. Arabic-script tokens containing a Sindhi-specific letter (`ڄ ڃ ڇ ڏ ڊ ڍ ڌ ڙ ڻ ٺ ٽ ٿ ڀ ڦ ڪ ڳ ڱ ھ`) become `SND`. Otherwise, a token in the Sindhi lexicon becomes `SND`; all remaining Arabic-script tokens become `URD`.
3. Latin-script tokens are lowercased. Words in the Roman Urdu lexicon become `ROM`, words in the English lexicon become `ENG`, and words in neither become `OTHER`.
4. Words present in both the English and Roman Urdu lexicons (`do`, `hi`, `in`, `is`, `the`, `to`, `us`) are resolved using the nearest non-`OTHER` neighbour tag, falling back to `ENG`.

### Known limitations

- Roman Urdu spelling is highly variable, so lexicon coverage and accuracy vary by writer.
- The lexicons are intentionally small starting points; they need to grow from annotated data.
- Tagging is lexical and rule-based, with no trained model in V1.
- English loanwords used inside Roman Urdu are tagged `ENG` when they match the English lexicon.
- The Sindhi-specific-letter rule can over-tag Urdu words that contain `ھ`.
- The sample corpus is synthetic and is not a gold/annotated evaluation set.

## Data note

The bundled sample corpus is synthetic and exists only for demonstrations. Real chat or social-media data must be anonymised before use, and exports may still contain sensitive text. The app runs locally and has no accounts, database, or cloud storage.

## Research Note

### Research question

### Data

### Method

### Results

### Discussion

### Limitations

## Roadmap

- Roman Sindhi support
- Custom lexicon upload
- A simple ML tagger trained on the gold set
- A Praat-style audio module
- PDF report export
