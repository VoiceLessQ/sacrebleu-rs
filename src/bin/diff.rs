//! Differential harness binary for `difftest.py`. Reads metric test cases from stdin and prints
//! the score and statistics so the Python sacrebleu oracle can be compared against.
//!
//! Protocol (all lines `\n`-separated, trailing `\r` tolerated):
//!   line 1: number of cases C
//!   per case the header's first field selects the metric:
//!     bleu: bleu <TAB> smooth <TAB> lc(0/1) <TAB> eff(0/1) <TAB> tokenize <TAB> num_refs <TAB> num_segs
//!     chrf: chrf <TAB> lc(0/1) <TAB> whitespace(0/1) <TAB> eps(0/1) <TAB> word_order <TAB> num_refs <TAB> num_segs
//!   then num_segs hypothesis lines, then num_refs*num_segs reference lines (stream by stream).
//! Output, one line per case:
//!   bleu: score <TAB> sys_len <TAB> ref_len <TAB> "c.." <TAB> "t.." <TAB> bp <TAB> signature
//!   chrf: score <TAB> name <TAB> signature

use std::io::{self, Read, Write};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).expect("read stdin");

    // Tokenize-only mode (`diff spmtok`): one spm-tokenized line per input line, for measuring
    // SentencePiece piece parity against the Python package on a real corpus.
    if std::env::args().nth(1).as_deref() == Some("spmtok") {
        let model_path = std::env::var("SPM_MODEL").expect("SPM_MODEL not set");
        let sp = sacrebleu_rs::SentencePieceProcessor::open(&model_path).expect("load spm model");
        let mut out = String::new();
        for line in input.split('\n') {
            let line = line.strip_suffix('\r').unwrap_or(line);
            out.push_str(&sacrebleu_rs::encode_as_pieces(&sp, line));
            out.push('\n');
        }
        io::stdout().write_all(out.as_bytes()).expect("write stdout");
        return;
    }

    let mut lines = input.split('\n').map(|l| l.strip_suffix('\r').unwrap_or(l));

    let n_cases: usize = lines.next().unwrap_or("0").trim().parse().unwrap_or(0);
    let mut out = String::new();

    // The flores/spm SentencePiece model, loaded once if SPM_MODEL points at a .model file.
    let spm_model = std::env::var("SPM_MODEL").ok().and_then(|p| {
        sacrebleu_rs::SentencePieceProcessor::open(&p)
            .ok()
            .map(std::sync::Arc::new)
    });

    for _ in 0..n_cases {
        let header = lines.next().expect("missing header");
        let p: Vec<&str> = header.split('\t').collect();
        let metric = p[0];

        let read_corpus = |lines: &mut dyn Iterator<Item = &str>, num_refs: usize, num_segs: usize| {
            let hyps: Vec<String> = (0..num_segs).map(|_| lines.next().unwrap().to_string()).collect();
            let refs: Vec<Vec<String>> = (0..num_refs)
                .map(|_| (0..num_segs).map(|_| lines.next().unwrap().to_string()).collect())
                .collect();
            (hyps, refs)
        };

        if metric == "ter" {
            let normalized = p[1] == "1";
            let no_punct = p[2] == "1";
            let case_sensitive = p[3] == "1";
            let num_refs: usize = p[4].parse().unwrap();
            let num_segs: usize = p[5].parse().unwrap();
            let (hyps, refs) = read_corpus(&mut lines, num_refs, num_segs);

            let ter = sacrebleu_rs::Ter { normalized, no_punct, case_sensitive };
            let s = ter.corpus_score(&hyps, &refs);
            out.push_str(&format!(
                "{}\t{}\t{}\t{}\n",
                s.score, s.num_edits, s.ref_length, ter.signature(num_refs as i64)
            ));
        } else if metric == "chrf" {
            let lowercase = p[1] == "1";
            let whitespace = p[2] == "1";
            let eps = p[3] == "1";
            let word_order: usize = p[4].parse().unwrap();
            let num_refs: usize = p[5].parse().unwrap();
            let num_segs: usize = p[6].parse().unwrap();
            let (hyps, refs) = read_corpus(&mut lines, num_refs, num_segs);

            let chrf = sacrebleu_rs::Chrf {
                word_order,
                lowercase,
                whitespace,
                eps_smoothing: eps,
                ..sacrebleu_rs::Chrf::default()
            };
            let s = chrf.corpus_score(&hyps, &refs);
            out.push_str(&format!("{}\t{}\t{}\n", s.score, s.name(), chrf.signature(num_refs as i64)));
        } else {
            let smooth = p[1];
            let lowercase = p[2] == "1";
            let effective = p[3] == "1";
            let tokenize = p[4];
            let num_refs: usize = p[5].parse().unwrap();
            let num_segs: usize = p[6].parse().unwrap();
            let (hyps, refs) = read_corpus(&mut lines, num_refs, num_segs);

            let bleu = sacrebleu_rs::Bleu {
                lowercase,
                smooth_method: smooth.to_string(),
                smooth_value: None,
                max_ngram_order: sacrebleu_rs::MAX_NGRAM_ORDER,
                effective_order: effective,
                tokenize: tokenize.to_string(),
                spm_model: spm_model.clone(),
            };
            let s = bleu.corpus_score(&hyps, &refs);
            let counts = s.counts.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(" ");
            let totals = s.totals.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" ");
            out.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                s.score, s.sys_len, s.ref_len, counts, totals, s.bp, bleu.signature(num_refs as i64)
            ));
        }
    }

    io::stdout().write_all(out.as_bytes()).expect("write stdout");
}
