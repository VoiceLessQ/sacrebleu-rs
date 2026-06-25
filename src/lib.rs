//! A Rust port of [sacrebleu](https://github.com/mjpost/sacrebleu): reproducible
//! machine-translation metrics. This first slice covers **BLEU** with the `13a` tokenizer.
//!
//! Port target: the Python `sacrebleu` package (2.6.0). The deep core is score-faithful: the
//! `13a` tokenization, the integer n-gram sufficient statistics, the brevity penalty, the
//! smoothing methods, and the `my_log` floor all match the reference exactly. The surrounding API
//! is an idiomatic Rust translation rather than a line-for-line copy.
//!
//! Floating-point note: the final score goes through `exp`/`log`, whose last bit can differ
//! between Python's libm and Rust's. The integer statistics (counts, totals, lengths) are exact;
//! the score matches to well within any reported precision.

use regex::Regex;
use std::collections::HashMap;
use std::sync::OnceLock;

/// The default maximum n-gram order.
pub const MAX_NGRAM_ORDER: usize = 4;

/// The sacrebleu version embedded in reproducibility signatures (the pinned oracle version).
pub const VERSION: &str = "2.6.0";

// --- tokenizer: 13a (deep core, byte-exact) --------------------------------------------------

fn regexp_rules() -> &'static [(Regex, &'static str)] {
    static RULES: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    RULES.get_or_init(|| {
        vec![
            // language-dependent part (assuming Western languages): wrap punctuation in spaces.
            (
                Regex::new(r"([\x20-\x26\x28-\x2b\x2f\x3a-\x40\x5b-\x60\x7b-\x7e])").unwrap(),
                " $1 ",
            ),
            // tokenize period and comma unless preceded by a digit
            (Regex::new(r"([^0-9])([\.,])").unwrap(), "$1 $2 "),
            // tokenize period and comma unless followed by a digit
            (Regex::new(r"([\.,])([^0-9])").unwrap(), " $1 $2"),
            // tokenize dash when preceded by a digit
            (Regex::new(r"([0-9])(-)").unwrap(), "$1 $2 "),
        ]
    })
}

/// Port of `TokenizerRegexp`: the shared post-tokenizer for `13a`.
fn tokenize_regexp(line: &str) -> String {
    let mut s = line.to_string();
    for (re, repl) in regexp_rules() {
        s = re.replace_all(&s, *repl).into_owned();
    }
    // no leading/trailing spaces, single space within words
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Port of `Tokenizer13a`: the mteval-v13a-equivalent tokenizer used by WMT.
pub fn tokenize_13a(line: &str) -> String {
    let mut line = line.replace("<skipped>", "");
    line = line.replace("-\n", "");
    line = line.replace('\n', " ");
    if line.contains('&') {
        line = line.replace("&quot;", "\"");
        line = line.replace("&amp;", "&");
        line = line.replace("&lt;", "<");
        line = line.replace("&gt;", ">");
    }
    tokenize_regexp(&format!(" {line} "))
}

fn intl_rules() -> &'static [(Regex, &'static str)] {
    static RULES: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    RULES.get_or_init(|| {
        vec![
            // separate punctuation preceded by a non-digit
            (Regex::new(r"(\P{N})(\p{P})").unwrap(), "$1 $2 "),
            // separate punctuation followed by a non-digit
            (Regex::new(r"(\p{P})(\P{N})").unwrap(), " $1 $2"),
            // separate symbols
            (Regex::new(r"(\p{S})").unwrap(), " $1 "),
        ]
    })
}

/// Port of `TokenizerV14International` (mteval-v14): split on Unicode punctuation and symbols,
/// except a punctuation mark sitting between two digits (e.g. a decimal separator).
pub fn tokenize_intl(line: &str) -> String {
    let mut s = line.to_string();
    for (re, repl) in intl_rules() {
        s = re.replace_all(&s, *repl).into_owned();
    }
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Port of `TokenizerChar`: one token per Unicode scalar value.
pub fn tokenize_char(line: &str) -> String {
    line.chars().map(|c| c.to_string()).collect::<Vec<_>>().join(" ")
}

/// Dispatch by tokenizer name (`13a`, `intl`, `char`, `none`). Unknown names fall back to `13a`.
fn apply_tokenizer(name: &str, line: &str) -> String {
    match name {
        "intl" => tokenize_intl(line),
        "char" => tokenize_char(line),
        "none" => line.to_string(),
        _ => tokenize_13a(line),
    }
}

// --- n-gram helpers (deep core) --------------------------------------------------------------

/// Port of `extract_all_word_ngrams`: counts of every n-gram (1..=max_order) plus the token count.
fn extract_all_word_ngrams(line: &str, max_order: usize) -> (HashMap<Vec<String>, i64>, usize) {
    let tokens: Vec<String> = line.split_whitespace().map(|s| s.to_string()).collect();
    let mut counts: HashMap<Vec<String>, i64> = HashMap::new();
    for n in 1..=max_order {
        if tokens.len() >= n {
            for i in 0..=tokens.len() - n {
                *counts.entry(tokens[i..i + n].to_vec()).or_insert(0) += 1;
            }
        }
    }
    (counts, tokens.len())
}

/// Merged reference n-grams (max count across refs) and the per-reference lengths.
struct RefInfo {
    ngrams: HashMap<Vec<String>, i64>,
    lens: Vec<usize>,
}

/// Port of `_extract_reference_info`.
fn extract_reference_info(refs: &[String], max_order: usize) -> RefInfo {
    let mut ngrams: HashMap<Vec<String>, i64> = HashMap::new();
    let mut lens = Vec::with_capacity(refs.len());
    for (idx, r) in refs.iter().enumerate() {
        let (this, len) = extract_all_word_ngrams(r, max_order);
        lens.push(len);
        if idx == 0 {
            ngrams = this;
        } else {
            for (ng, count) in this {
                let e = ngrams.entry(ng).or_insert(0);
                *e = (*e).max(count);
            }
        }
    }
    RefInfo { ngrams, lens }
}

/// Port of `_get_closest_ref_len`: closest reference length, ties broken toward the shorter.
fn closest_ref_len(hyp_len: usize, ref_lens: &[usize]) -> usize {
    let mut closest_diff: i64 = -1;
    let mut closest_len: i64 = -1;
    for &rl in ref_lens {
        let diff = (hyp_len as i64 - rl as i64).abs();
        if closest_diff == -1 || diff < closest_diff {
            closest_diff = diff;
            closest_len = rl as i64;
        } else if diff == closest_diff && (rl as i64) < closest_len {
            closest_len = rl as i64;
        }
    }
    closest_len.max(0) as usize
}

/// Port of `_compute_segment_statistics`: `[hyp_len, ref_len, correct.., total..]`.
fn segment_statistics(hyp: &str, refinfo: &RefInfo, max_order: usize) -> Vec<i64> {
    let (hyp_ngrams, hyp_len) = extract_all_word_ngrams(hyp, max_order);
    let ref_len = closest_ref_len(hyp_len, &refinfo.lens);

    let mut correct = vec![0i64; max_order];
    let mut total = vec![0i64; max_order];
    for (ng, count) in &hyp_ngrams {
        let n = ng.len() - 1;
        total[n] += count;
        if let Some(&rc) = refinfo.ngrams.get(ng) {
            correct[n] += (*count).min(rc);
        }
    }

    let mut stats = Vec::with_capacity(2 + 2 * max_order);
    stats.push(hyp_len as i64);
    stats.push(ref_len as i64);
    stats.extend_from_slice(&correct);
    stats.extend_from_slice(&total);
    stats
}

// --- scoring (deep core) ---------------------------------------------------------------------

/// Port of `my_log`: `log`, floored to a very low number at zero.
fn my_log(num: f64) -> f64 {
    if num == 0.0 { -9999999999.0 } else { num.ln() }
}

fn smooth_default(method: &str) -> Option<f64> {
    match method {
        "floor" => Some(0.1),
        "add-k" => Some(1.0),
        _ => None, // none, exp
    }
}

/// The result of a BLEU computation. Mirrors sacrebleu's `BLEUScore`.
#[derive(Debug, Clone)]
pub struct BleuScore {
    pub score: f64,
    pub counts: Vec<i64>,
    pub totals: Vec<i64>,
    pub precisions: Vec<f64>,
    pub bp: f64,
    pub sys_len: i64,
    pub ref_len: i64,
}

impl BleuScore {
    /// `prec0/prec1/.../ (BP = .. ratio = .. hyp_len = .. ref_len = ..)`, like sacrebleu's verbose.
    pub fn verbose(&self) -> String {
        let prec_str = self
            .precisions
            .iter()
            .map(|p| format!("{p:.1}"))
            .collect::<Vec<_>>()
            .join("/");
        let ratio = if self.ref_len != 0 {
            self.sys_len as f64 / self.ref_len as f64
        } else {
            0.0
        };
        format!(
            "{prec_str} (BP = {:.3} ratio = {:.3} hyp_len = {} ref_len = {})",
            self.bp, ratio, self.sys_len, self.ref_len
        )
    }
}

/// Port of `BLEU.compute_bleu`: the score from its sufficient statistics, with smoothing.
#[allow(clippy::too_many_arguments)]
pub fn compute_bleu(
    mut correct: Vec<i64>,
    mut total: Vec<i64>,
    sys_len: i64,
    ref_len: i64,
    smooth_method: &str,
    smooth_value: Option<f64>,
    effective_order: bool,
    max_ngram_order: usize,
) -> BleuScore {
    let smooth_value = smooth_value.or_else(|| smooth_default(smooth_method));

    // Brevity penalty.
    let mut bp = 1.0;
    if sys_len < ref_len {
        bp = if sys_len > 0 {
            (1.0 - ref_len as f64 / sys_len as f64).exp()
        } else {
            0.0
        };
    }

    let mut precisions = vec![0.0f64; max_ngram_order];

    // Early stop if there are no matches at all.
    if correct.iter().all(|&c| c == 0) {
        return BleuScore { score: 0.0, counts: correct, totals: total, precisions, bp, sys_len, ref_len };
    }

    let mut smooth_mteval = 1.0;
    let mut eff_order = max_ngram_order;
    for n in 1..=precisions.len() {
        if smooth_method == "add-k" && n > 1 {
            let sv = smooth_value.unwrap_or(1.0) as i64;
            correct[n - 1] += sv;
            total[n - 1] += sv;
        }
        if total[n - 1] == 0 {
            break;
        }
        if effective_order {
            eff_order = n;
        }
        if correct[n - 1] == 0 {
            if smooth_method == "exp" {
                smooth_mteval *= 2.0;
                precisions[n - 1] = 100.0 / (smooth_mteval * total[n - 1] as f64);
            } else if smooth_method == "floor" {
                precisions[n - 1] = 100.0 * smooth_value.unwrap_or(0.1) / total[n - 1] as f64;
            }
        } else {
            precisions[n - 1] = 100.0 * correct[n - 1] as f64 / total[n - 1] as f64;
        }
    }

    let log_sum: f64 = precisions[..eff_order].iter().map(|&p| my_log(p)).sum();
    let score = bp * (log_sum / eff_order as f64).exp();

    BleuScore { score, counts: correct, totals: total, precisions, bp, sys_len, ref_len }
}

// --- public metric ---------------------------------------------------------------------------

/// The BLEU metric. Mirrors sacrebleu's `BLEU` (this slice uses the `13a` tokenizer).
#[derive(Debug, Clone)]
pub struct Bleu {
    pub lowercase: bool,
    pub smooth_method: String,
    pub smooth_value: Option<f64>,
    pub max_ngram_order: usize,
    pub effective_order: bool,
    /// Tokenizer name: `13a` (default), `intl`, `char`, or `none`.
    pub tokenize: String,
}

impl Default for Bleu {
    fn default() -> Self {
        Bleu {
            lowercase: false,
            smooth_method: "exp".to_string(),
            smooth_value: None,
            max_ngram_order: MAX_NGRAM_ORDER,
            effective_order: false,
            tokenize: "13a".to_string(),
        }
    }
}

impl Bleu {
    fn preprocess(&self, sent: &str) -> String {
        let lowered;
        let s = if self.lowercase {
            lowered = sent.to_lowercase();
            lowered.as_str()
        } else {
            sent
        };
        apply_tokenizer(&self.tokenize, s.trim_end())
    }

    /// Corpus-level BLEU. `refs` is a list of reference *streams* (sacrebleu layout): `refs[r][i]`
    /// is the r-th reference of the i-th hypothesis.
    pub fn corpus_score(&self, hyps: &[String], refs: &[Vec<String>]) -> BleuScore {
        let mo = self.max_ngram_order;
        let mut agg = vec![0i64; 2 + 2 * mo];
        for (i, hyp) in hyps.iter().enumerate() {
            let refs_for_seg: Vec<String> =
                refs.iter().map(|stream| self.preprocess(&stream[i])).collect();
            let refinfo = extract_reference_info(&refs_for_seg, mo);
            let hyp_tok = self.preprocess(hyp);
            let stats = segment_statistics(&hyp_tok, &refinfo, mo);
            for (a, s) in agg.iter_mut().zip(stats) {
                *a += s;
            }
        }
        compute_bleu(
            agg[2..2 + mo].to_vec(),
            agg[2 + mo..].to_vec(),
            agg[0],
            agg[1],
            &self.smooth_method,
            self.smooth_value,
            self.effective_order,
            mo,
        )
    }

    /// Sentence-level BLEU against one or more references.
    pub fn sentence_score(&self, hyp: &str, refs: &[String]) -> BleuScore {
        let streams: Vec<Vec<String>> = refs.iter().map(|r| vec![r.clone()]).collect();
        self.corpus_score(&[hyp.to_string()], &streams)
    }

    /// The reproducibility signature, e.g.
    /// `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`. Pass the number of reference
    /// streams (or -1 for a variable number), as sacrebleu reports `nrefs:var` in that case.
    pub fn signature(&self, num_refs: i64) -> String {
        let nrefs = if num_refs == -1 { "var".to_string() } else { num_refs.to_string() };
        let case = if self.lowercase { "lc" } else { "mixed" };
        let eff = if self.effective_order { "yes" } else { "no" };
        let tok = tokenizer_signature(&self.tokenize);
        let smooth = smooth_signature(&self.smooth_method, self.smooth_value);
        format!("nrefs:{nrefs}|case:{case}|eff:{eff}|tok:{tok}|smooth:{smooth}|version:{VERSION}")
    }
}

/// The signature name for a tokenizer (matches sacrebleu's `signature()` methods).
fn tokenizer_signature(name: &str) -> &str {
    match name {
        "intl" => "intl",
        "char" => "char",
        "none" => "none",
        _ => "13a",
    }
}

/// The `smooth` field of the signature: the method, plus `[value]` for `floor`/`add-k`.
fn smooth_signature(method: &str, value: Option<f64>) -> String {
    match smooth_default(method) {
        Some(def) => format!("{method}[{:.2}]", value.unwrap_or(def)),
        None => method.to_string(),
    }
}

// --- chrF / chrF++ ---------------------------------------------------------------------------

/// Character n-grams (orders 1..=`max_order`) of a line. Whitespace is removed first unless
/// `include_ws`. Port of `extract_all_char_ngrams`.
fn extract_all_char_ngrams(line: &str, max_order: usize, include_ws: bool) -> Vec<HashMap<String, i64>> {
    let chars: Vec<char> = if include_ws {
        line.chars().collect()
    } else {
        line.split_whitespace().collect::<Vec<_>>().join("").chars().collect()
    };
    let mut out = Vec::with_capacity(max_order);
    for n in 1..=max_order {
        let mut c: HashMap<String, i64> = HashMap::new();
        if chars.len() >= n {
            for i in 0..=chars.len() - n {
                let ng: String = chars[i..i + n].iter().collect();
                *c.entry(ng).or_insert(0) += 1;
            }
        }
        out.push(c);
    }
    out
}

/// Word n-grams of order `n` over `words`, keyed by the space-joined token. Port of
/// `extract_word_ngrams`.
fn extract_word_ngrams(words: &[String], n: usize) -> HashMap<String, i64> {
    let mut c: HashMap<String, i64> = HashMap::new();
    if words.len() >= n {
        for i in 0..=words.len() - n {
            *c.entry(words[i..i + n].join(" ")).or_insert(0) += 1;
        }
    }
    c
}

const CHRF_PUNCTS: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

/// Port of `CHRF._remove_punctuation`: split punctuation off the ends of multi-char words.
fn remove_punctuation(sent: &str) -> Vec<String> {
    let mut out = Vec::new();
    for w in sent.split_whitespace() {
        let cs: Vec<char> = w.chars().collect();
        if cs.len() == 1 {
            out.push(w.to_string());
        } else if CHRF_PUNCTS.contains(cs[cs.len() - 1]) {
            out.push(cs[..cs.len() - 1].iter().collect());
            out.push(cs[cs.len() - 1].to_string());
        } else if CHRF_PUNCTS.contains(cs[0]) {
            out.push(cs[0].to_string());
            out.push(cs[1..].iter().collect());
        } else {
            out.push(w.to_string());
        }
    }
    out
}

/// Port of `CHRF._get_match_statistics`: `[hyp_count, ref_count, match_count]` for one order.
fn chrf_match_stats(hyp: &HashMap<String, i64>, refc: &HashMap<String, i64>) -> [i64; 3] {
    let mut match_count = 0i64;
    let mut hyp_count = 0i64;
    for (ng, &count) in hyp {
        hyp_count += count;
        if let Some(&rc) = refc.get(ng) {
            match_count += count.min(rc);
        }
    }
    let ref_total: i64 = refc.values().sum();
    [if refc.is_empty() { 0 } else { hyp_count }, ref_total, match_count]
}

/// Port of `CHRF._compute_f_score`.
fn chrf_f_score(stats: &[i64], order: usize, beta: i64, eps_smoothing: bool) -> f64 {
    let eps = 1e-16;
    let mut score = 0.0;
    let mut effective_order = 0usize;
    let factor = (beta * beta) as f64;
    let mut avg_prec = 0.0;
    let mut avg_rec = 0.0;

    for i in 0..order {
        let n_hyp = stats[3 * i] as f64;
        let n_ref = stats[3 * i + 1] as f64;
        let n_match = stats[3 * i + 2] as f64;

        let prec = if n_hyp > 0.0 { n_match / n_hyp } else { eps };
        let rec = if n_ref > 0.0 { n_match / n_ref } else { eps };

        let denom = factor * prec + rec;
        score += if denom > 0.0 { (1.0 + factor) * prec * rec / denom } else { eps };

        if n_hyp > 0.0 && n_ref > 0.0 {
            avg_prec += prec;
            avg_rec += rec;
            effective_order += 1;
        }
    }

    if eps_smoothing {
        return 100.0 * score / order as f64;
    }

    if effective_order == 0 {
        avg_prec = 0.0;
        avg_rec = 0.0;
    } else {
        avg_prec /= effective_order as f64;
        avg_rec /= effective_order as f64;
    }

    if avg_prec + avg_rec != 0.0 {
        let s = (1.0 + factor) * avg_prec * avg_rec / (factor * avg_prec + avg_rec);
        100.0 * s
    } else {
        0.0
    }
}

/// The result of a chrF computation. Mirrors sacrebleu's `CHRFScore`.
#[derive(Debug, Clone)]
pub struct ChrfScore {
    pub score: f64,
    pub char_order: usize,
    pub word_order: usize,
    pub beta: i64,
}

impl ChrfScore {
    /// `chrF{beta}` plus a `+` per word order, e.g. `chrF2` or `chrF2++`.
    pub fn name(&self) -> String {
        format!("chrF{}{}", self.beta, "+".repeat(self.word_order))
    }
}

/// The chrF / chrF++ metric. Mirrors sacrebleu's `CHRF`.
#[derive(Debug, Clone)]
pub struct Chrf {
    pub char_order: usize,
    pub word_order: usize,
    pub beta: i64,
    pub lowercase: bool,
    pub whitespace: bool,
    pub eps_smoothing: bool,
}

impl Default for Chrf {
    fn default() -> Self {
        Chrf {
            char_order: 6,
            word_order: 0,
            beta: 2,
            lowercase: false,
            whitespace: false,
            eps_smoothing: false,
        }
    }
}

impl Chrf {
    fn order(&self) -> usize {
        self.char_order + self.word_order
    }

    fn segment_ngrams(&self, sent: &str) -> Vec<HashMap<String, i64>> {
        let mut v = extract_all_char_ngrams(sent, self.char_order, self.whitespace);
        if self.word_order > 0 {
            let words = remove_punctuation(sent);
            for n in 1..=self.word_order {
                v.push(extract_word_ngrams(&words, n));
            }
        }
        v
    }

    fn preprocess(&self, sent: &str) -> String {
        if self.lowercase { sent.to_lowercase() } else { sent.to_string() }
    }

    /// Corpus-level chrF. `refs` uses sacrebleu's layout (`refs[r][i]` is the r-th reference of
    /// the i-th hypothesis); the best-matching reference is chosen per segment.
    pub fn corpus_score(&self, hyps: &[String], refs: &[Vec<String>]) -> ChrfScore {
        let order = self.order();
        let mut agg = vec![0i64; 3 * order];

        for (i, hyp) in hyps.iter().enumerate() {
            let hyp_ng = self.segment_ngrams(&self.preprocess(hyp));

            let mut best_stats: Vec<i64> = Vec::new();
            let mut best_f = -1.0;
            for stream in refs {
                let ref_ng = self.segment_ngrams(&self.preprocess(&stream[i]));
                let mut stats = Vec::with_capacity(3 * order);
                for (h, r) in hyp_ng.iter().zip(ref_ng.iter()) {
                    stats.extend_from_slice(&chrf_match_stats(h, r));
                }
                let f = chrf_f_score(&stats, order, self.beta, self.eps_smoothing);
                if f > best_f {
                    best_f = f;
                    best_stats = stats;
                }
            }
            for (a, s) in agg.iter_mut().zip(best_stats) {
                *a += s;
            }
        }

        let score = chrf_f_score(&agg, order, self.beta, self.eps_smoothing);
        ChrfScore { score, char_order: self.char_order, word_order: self.word_order, beta: self.beta }
    }

    /// Sentence-level chrF against one or more references.
    pub fn sentence_score(&self, hyp: &str, refs: &[String]) -> ChrfScore {
        let streams: Vec<Vec<String>> = refs.iter().map(|r| vec![r.clone()]).collect();
        self.corpus_score(&[hyp.to_string()], &streams)
    }

    /// The reproducibility signature, e.g.
    /// `nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0`.
    pub fn signature(&self, num_refs: i64) -> String {
        let nrefs = if num_refs == -1 { "var".to_string() } else { num_refs.to_string() };
        let case = if self.lowercase { "lc" } else { "mixed" };
        let eff = if !self.eps_smoothing { "yes" } else { "no" };
        let space = if self.whitespace { "yes" } else { "no" };
        format!(
            "nrefs:{nrefs}|case:{case}|eff:{eff}|nc:{}|nw:{}|space:{space}|version:{VERSION}",
            self.char_order, self.word_order
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_basics() {
        assert_eq!(tokenize_13a("Hello, World!"), "Hello , World !");
        assert_eq!(tokenize_13a("It's a test."), "It's a test .");
        // comma between digits is kept; dash after a digit splits
        assert_eq!(tokenize_13a("1,000 and 7-up"), "1,000 and 7 - up");
        // entities are unescaped, then quotes are split off as their own tokens
        assert_eq!(tokenize_13a("a &amp; b &quot;c&quot;"), "a & b \" c \"");
    }

    #[test]
    fn perfect_match_is_100() {
        let b = Bleu::default();
        let s = b.corpus_score(
            &["the cat sat on the mat".to_string()],
            &[vec!["the cat sat on the mat".to_string()]],
        );
        assert!((s.score - 100.0).abs() < 1e-6);
        assert_eq!(s.sys_len, 6);
        assert_eq!(s.ref_len, 6);
    }

    #[test]
    fn no_match_is_zero() {
        let b = Bleu::default();
        let s = b.corpus_score(
            &["completely different words here".to_string()],
            &[vec!["nothing alike at all".to_string()]],
        );
        assert_eq!(s.score, 0.0);
    }

    #[test]
    fn other_tokenizers() {
        assert_eq!(tokenize_char("abc"), "a b c");
        assert_eq!(tokenize_char("我爱"), "我 爱");
        assert_eq!(tokenize_intl("hello, world!"), "hello , world !");
        assert_eq!(tokenize_intl("3.14"), "3.14"); // a dot between digits stays
        assert_eq!(tokenize_intl("a+b"), "a + b"); // '+' is a symbol
        assert_eq!(apply_tokenizer("none", "a , b"), "a , b"); // identity
    }

    #[test]
    fn signatures() {
        assert_eq!(
            Bleu::default().signature(1),
            "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0"
        );
        let b = Bleu {
            lowercase: true,
            effective_order: true,
            smooth_method: "floor".to_string(),
            tokenize: "intl".to_string(),
            ..Bleu::default()
        };
        assert_eq!(
            b.signature(2),
            "nrefs:2|case:lc|eff:yes|tok:intl|smooth:floor[0.10]|version:2.6.0"
        );
    }

    #[test]
    fn chrf_basics() {
        let c = Chrf::default();
        let s = c.corpus_score(&["the cat".to_string()], &[vec!["the cat".to_string()]]);
        assert!((s.score - 100.0).abs() < 1e-9);
        assert_eq!(s.name(), "chrF2");
        assert_eq!(
            c.signature(1),
            "nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0"
        );

        let cpp = Chrf { word_order: 2, ..Chrf::default() };
        let s2 = cpp.corpus_score(&["the cat".to_string()], &[vec!["the cat".to_string()]]);
        assert_eq!(s2.name(), "chrF2++");
        assert!((s2.score - 100.0).abs() < 1e-9);
    }
}
