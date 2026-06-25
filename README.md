# sacrebleu-rs

[![CI](https://github.com/VoiceLessQ/sacrebleu-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/VoiceLessQ/sacrebleu-rs/actions/workflows/ci.yml)

A Rust port of [sacrebleu](https://github.com/mjpost/sacrebleu): reproducible machine-translation
metrics. It is score-faithful to the Python package and verified by differential testing.

Status: BLEU (tokenizers `13a`, `intl`, `char`, `none`, and `spm`/`flores` via SentencePiece),
chrF/chrF++, and TER are implemented, each with its reproducibility signature. The `ja-mecab` and
`ko-mecab` tokenizers (which need MeCab) and TER's Asian normalization are not included.

## Usage

```rust
use sacrebleu_rs::Bleu;

let bleu = Bleu::default();
let score = bleu.corpus_score(
    &["the cat sat on the mat".to_string()],
    &[vec!["the cat sat on the mat".to_string()]],
);
println!("{:.2}", score.score); // 100.00

// Sentence level (enable effective_order for short sentences, as sacrebleu recommends).
let s = Bleu { effective_order: true, ..Bleu::default() }
    .sentence_score("the cat sat", &["the cat sat on the mat".to_string()]);
```

```rust
use sacrebleu_rs::{Chrf, Ter};

let chrf = Chrf::default(); // word_order: 2 gives chrF++
let ter = Ter::default();
let h = ["the cat sat on the mat".to_string()];
let r = [vec!["the cat sat on the mat".to_string()]];
println!("{:.2}", chrf.corpus_score(&h, &r).score);
println!("{:.2}", ter.corpus_score(&h, &r).score);
```

For the `spm`/`flores` tokenizers, load a SentencePiece model and hand BLEU the processor:

```rust
use std::sync::Arc;
use sacrebleu_rs::{Bleu, SentencePieceProcessor};

let sp = Arc::new(SentencePieceProcessor::open("flores200_sacrebleu.model")?);
let bleu = Bleu { tokenize: "flores200".into(), spm_model: Some(sp), ..Bleu::default() };
# Ok::<(), Box<dyn std::error::Error>>(())
```

References use sacrebleu's layout: `refs[r][i]` is the r-th reference of the i-th hypothesis.

## Fidelity

The deep core is score-faithful to sacrebleu 2.6.0: the tokenizations, the integer sufficient
statistics (BLEU n-gram counts and lengths, chrF match counts, TER edits and reference lengths),
the brevity penalty and smoothing, the chrF F-score, and the TER beam edit distance with shifts
all match the reference exactly, as do the reproducibility signatures.

BLEU's final score passes through `exp`/`log`, whose last bit can differ between Python's libm and
Rust's, so it is exact only to within floating-point tolerance; the integer statistics are exact.
chrF and TER use only `+`, `*`, `/`, so they match even more tightly. The surrounding API is an
idiomatic Rust translation rather than a line-for-line copy.

## Verification

Behavior is checked by differential testing against the Python `sacrebleu` package (2.6.0). A
matrix of corpora and options runs through this crate and `BLEU`, `CHRF`, and `TER`, comparing the
integer statistics and signatures for exact equality and each score to within 1e-9. The `flores200`
tokenizer is additionally checked piece-for-piece against `EncodeAsPieces` over thousands of lines
of English and Japanese.

## License

Licensed under the [MIT License](LICENSE-MIT). The upstream sacrebleu is Apache-2.0; this is an
independent reimplementation.
