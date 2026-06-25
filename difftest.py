r"""Differential test: sacrebleu-rs vs the Python `sacrebleu` package (the oracle).

Runs matrices of corpora x options through both this crate (via the `diff` binary) and sacrebleu's
`BLEU` and `CHRF`, checking the integer sufficient statistics and signatures agree exactly and the
final score agrees to within floating-point tolerance.

Run from the sacrebleu-rs/ folder with the in-tree Python (after `cargo build`):
    ..\..\python\python\python.exe difftest.py
"""

import os
import subprocess
import sys

from sacrebleu.metrics import BLEU, CHRF, TER
from sacrebleu.utils import SACREBLEU_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
RUST_BIN = os.path.join(HERE, "target", "debug", "diff.exe" if os.name == "nt" else "diff")
# The flores200 SentencePiece model (sacrebleu downloads it on first use of tokenize="flores200").
FLORES_MODEL = os.path.join(SACREBLEU_DIR, "models", "flores200sacrebleuspm")

# Each corpus: (hypotheses, references) where references is a list of reference streams.
CORPORA = [
    (["the cat sat on the mat"], [["the cat sat on the mat"]]),
    (["the cat sat on the mat"], [["a cat was sitting on the mat"]]),
    (
        ["the quick brown fox", "jumps over the lazy dog", "hello world"],
        [["the quick brown fox", "jumped over a lazy dog", "hello there world"]],
    ),
    (["the cat is on the mat"], [["the cat sat on the mat"], ["there is a cat on the mat"]]),
    (
        ["In 2024, the well-known author said &quot;hello&quot;."],
        [["In 2024 the well-known author said \"hello\"."]],
    ),
    (["good", "very good"], [["good", "really very good indeed"]]),
    (["the cat sat on the mat and slept"], [["the cat sat on the mat"]]),
    (["the the the the the the"], [["the cat sat on the mat"]]),
    (["The Cat SAT on the Mat"], [["the cat sat on the mat"]]),
    (["Café déjà-vu: 3.14 € costs $5, n° 1!"], [["Café déjà vu 3.14 € is $5 number 1"]]),
    (["我爱北京天安门"], [["我爱北京"]]),
    (["the cat sat ."], [["the cat sat on the mat ."]]),
]

SMOOTHS = ["exp", "floor", "none", "add-k"]
BLEU_OPTS = [(False, False), (True, False), (False, True)]  # (lowercase, effective_order)
TOKS = ["13a", "intl", "char", "none"]

# chrF: (word_order, lowercase, whitespace, eps_smoothing)
CHRF_OPTS = [
    (wo, lc, ws, eps)
    for wo in (0, 2)
    for lc in (False, True)
    for ws in (False, True)
    for eps in (False, True)
]

# TER: (normalized, no_punct, case_sensitive)
TER_OPTS = [
    (norm, npunct, cs)
    for norm in (False, True)
    for npunct in (False, True)
    for cs in (False, True)
]


def build_input(cases):
    lines = [str(len(cases))]
    for c in cases:
        if c["kind"] == "bleu":
            lines.append(f"bleu\t{c['smooth']}\t{int(c['lc'])}\t{int(c['eff'])}\t{c['tok']}\t{len(c['refs'])}\t{len(c['hyps'])}")
        elif c["kind"] == "ter":
            lines.append(f"ter\t{int(c['norm'])}\t{int(c['npunct'])}\t{int(c['cs'])}\t{len(c['refs'])}\t{len(c['hyps'])}")
        else:
            lines.append(f"chrf\t{int(c['lc'])}\t{int(c['ws'])}\t{int(c['eps'])}\t{c['wo']}\t{len(c['refs'])}\t{len(c['hyps'])}")
        lines.extend(c["hyps"])
        for stream in c["refs"]:
            lines.extend(stream)
    return "\n".join(lines)


def main():
    if not os.path.exists(RUST_BIN):
        sys.exit(f"missing {RUST_BIN} - run `cargo build` first")

    # Ensure the flores200 SentencePiece model is present (downloads on first construction).
    # If it can't be fetched, the spm cases are skipped rather than failing the run.
    if not os.path.exists(FLORES_MODEL):
        try:
            BLEU(tokenize="flores200")
        except Exception as e:  # noqa: BLE001
            print(f"(flores200 model unavailable, skipping spm cases: {e})")

    cases = []
    for (hyps, refs) in CORPORA:
        for smooth in SMOOTHS:
            for (lc, eff) in BLEU_OPTS:
                for tok in TOKS:
                    cases.append({"kind": "bleu", "hyps": hyps, "refs": refs,
                                  "smooth": smooth, "lc": lc, "eff": eff, "tok": tok})
        for (wo, lc, ws, eps) in CHRF_OPTS:
            cases.append({"kind": "chrf", "hyps": hyps, "refs": refs,
                          "wo": wo, "lc": lc, "ws": ws, "eps": eps})
        for (norm, npunct, cs) in TER_OPTS:
            cases.append({"kind": "ter", "hyps": hyps, "refs": refs,
                          "norm": norm, "npunct": npunct, "cs": cs})
        # spm/flores200 BLEU (needs the downloaded SentencePiece model)
        if os.path.exists(FLORES_MODEL):
            for lc in (False, True):
                cases.append({"kind": "bleu", "hyps": hyps, "refs": refs,
                              "smooth": "exp", "lc": lc, "eff": False, "tok": "flores200"})

    env = dict(os.environ, SPM_MODEL=FLORES_MODEL)
    proc = subprocess.run(
        [RUST_BIN], input=build_input(cases), capture_output=True, text=True,
        encoding="utf-8", env=env,
    )
    if proc.returncode != 0:
        sys.exit(f"rust diff binary failed:\n{proc.stderr}")

    rust_lines = proc.stdout.split("\n")
    mismatches = []
    for c, rline in zip(cases, rust_lines):
        f = rline.split("\t")
        if c["kind"] == "bleu":
            b = BLEU(smooth_method=c["smooth"], lowercase=c["lc"], effective_order=c["eff"],
                     tokenize=c["tok"], force=True)
            exp = b.corpus_score(c["hyps"], c["refs"])
            r_score, r_sys, r_ref = float(f[0]), int(f[1]), int(f[2])
            r_counts = [int(x) for x in f[3].split()]
            r_totals = [int(x) for x in f[4].split()]
            r_sig = f[6]
            ok = (r_sys == exp.sys_len and r_ref == exp.ref_len
                  and r_counts == list(exp.counts) and r_totals == list(exp.totals)
                  and abs(r_score - exp.score) < 1e-9 and r_sig == str(b.get_signature()))
            if not ok:
                mismatches.append(("bleu", c, exp.score, str(b.get_signature()), r_score, r_sig,
                                   list(exp.counts), list(exp.totals), r_counts, r_totals))
        elif c["kind"] == "ter":
            t = TER(normalized=c["norm"], no_punct=c["npunct"], case_sensitive=c["cs"])
            exp = t.corpus_score(c["hyps"], c["refs"])
            r_score, r_edits, r_reflen, r_sig = float(f[0]), int(f[1]), float(f[2]), f[3]
            ok = (abs(r_score - exp.score) < 1e-9 and r_edits == exp.num_edits
                  and abs(r_reflen - exp.ref_length) < 1e-9 and r_sig == str(t.get_signature()))
            if not ok:
                mismatches.append(("ter", c, exp.score, str(t.get_signature()), r_score, r_sig,
                                   exp.num_edits, r_edits, exp.ref_length, r_reflen))
        else:
            ch = CHRF(word_order=c["wo"], lowercase=c["lc"], whitespace=c["ws"], eps_smoothing=c["eps"])
            exp = ch.corpus_score(c["hyps"], c["refs"])
            r_score, r_name, r_sig = float(f[0]), f[1], f[2]
            ok = (abs(r_score - exp.score) < 1e-9 and r_name == exp.name
                  and r_sig == str(ch.get_signature()))
            if not ok:
                mismatches.append(("chrf", c, exp.score, str(ch.get_signature()), r_score, r_sig,
                                   exp.name, r_name, None, None))

    if mismatches:
        print(f"{len(mismatches)} mismatches (of {len(cases)}):")
        for (kind, c, es, esig, rs, rsig, a, b, ra, rb) in mismatches[:20]:
            print(f"  {kind} {dict((k, v) for k, v in c.items() if k not in ('hyps', 'refs'))} hyp0={c['hyps'][0]!r}")
            print(f"      py : score={es:.6f} sig={esig} extra={a}/{b}")
            print(f"      rs : score={rs:.6f} sig={rsig} extra={ra}/{rb}")
        sys.exit("\nMISMATCHES FOUND.")

    n_bleu = sum(1 for c in cases if c["kind"] == "bleu")
    n_chrf = sum(1 for c in cases if c["kind"] == "chrf")
    n_ter = sum(1 for c in cases if c["kind"] == "ter")
    print(f"ALL MATCH - {n_bleu} BLEU + {n_chrf} chrF + {n_ter} TER computations agree with "
          f"sacrebleu (stats + signature exact, score < 1e-9).")


if __name__ == "__main__":
    main()
