# sacrebleu-rs

A Rust port of [sacrebleu](https://github.com/mjpost/sacrebleu): reproducible machine-translation
metrics. It is score-faithful to the Python package and verified by differential testing.

Status: BLEU with the `13a` tokenizer is implemented and verified. chrF, TER, the `intl`/`char`
tokenizers, and the reproducibility signature are planned.

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

References use sacrebleu's layout: `refs[r][i]` is the r-th reference of the i-th hypothesis.

## Fidelity

The deep core is score-faithful to sacrebleu 2.6.0: the `13a` tokenization, the integer n-gram
sufficient statistics (counts, totals, lengths), the brevity penalty, and the four smoothing
methods (`exp`, `floor`, `add-k`, `none`) all match the reference exactly.

The final score passes through `exp`/`log`, whose last bit can differ between Python's libm and
Rust's, so the score is exact only to within floating-point tolerance. The integer statistics are
exact, and the score matches to well within any reported precision. The surrounding API is an
idiomatic Rust translation rather than a line-for-line copy.

## Verification

Behavior is checked by differential testing against the Python `sacrebleu` package (2.6.0). A
matrix of corpora, smoothing methods, and options runs through both this crate and `BLEU`,
comparing the integer sufficient statistics for exact equality and the score to within 1e-9.

## License

Licensed under the [MIT License](LICENSE-MIT). The upstream sacrebleu is Apache-2.0; this is an
independent reimplementation.
