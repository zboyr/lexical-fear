# Language Models "Fear" Harmful Words: Causally Reducing a Lexically Triggered Driver of Over-Refusal

Code and data for the paper of the same title, accepted at the NeurIPS 2026 workshop *Foundations of Language Model Security* (FLMSec).

Aligned language models refuse harmless requests partly because of the words
those requests contain. The paper calls this the model's "fear" of harmful
words: an increase in refusal probability caused by harm-associated tokens
whose irrelevance to the request the model itself can verify. ("Fear" is a
narrative label, not a claim about model psychology.) On Qwen3.5-4B, inserting
one harm-topic word into a prompt the model otherwise tends to answer raises
its refusal rate by up to 33 points, even though the same model, asked in a
fresh chat, names the inserted word on demand. Two constructions locate the
reaction in late-layer hidden states at the generation boundary, the last
prompt tokens before the model starts answering. One is a refusal-supervised
linear probe compiled into a rank-one weight edit. The other is a label-free
crossed differencing of hidden states over harm/neutral word pairs. Writing
along the crossed-differencing direction on held-out prompt-word pairs moves
refusal in both directions, scales with dose, and beats KL-matched null
directions. Turning the direction down cuts OR-Bench Hard-1K over-refusal
under a pre-registered superiority gate. The safety and instruction-following
guards give a mixed verdict: every point estimate at the deployment doses
sits inside the registered 0.02 margin, but four of the six one-sided bounds
fall outside it, and refusal of genuinely toxic prompts slides slightly. In a
descriptive eight-model transfer, the activation write transfers across four
model families at one fixed dose. The rank-one weight edit at one fixed scale
does not, so its strength has to be calibrated per model.

> **Content warning.** This repository contains harmful and adversarial
> prompts (jailbreak-style requests, requests about weapons, drugs, fraud,
> sexual content, hate, self-harm and so on), a lexicon of harm-associated
> words, and model responses to these prompts. Some of those responses
> comply with harmful requests. The material is released only for research
> on language-model safety and over-refusal. Do not use it to train models to
> produce harmful content or to deploy it in any user-facing setting.

---

## Contents

1. [Getting the data](#getting-the-data)
2. [Repository layout](#repository-layout)
3. [Terms used in this README and in the code](#terms-used-in-this-readme-and-in-the-code)
4. [Quick check: headline numbers without GPUs](#quick-check-headline-numbers-without-gpus)
5. [Environment setup](#environment-setup)
6. [What is not included, and how to regenerate it](#what-is-not-included-and-how-to-regenerate-it)
7. [Reproduction guide](#reproduction-guide)
8. [Paper claims, figures and tables: result files and scripts](#paper-claims-figures-and-tables-result-files-and-scripts)
9. [Running the tests](#running-the-tests)
10. [Notes on the Slurm `.sbatch` files](#notes-on-the-slurm-sbatch-files)
11. [Citation](#citation)
12. [License](#license)

---

## Getting the data

The release is split across two places:

| Where | What |
|---|---|
| GitHub: `github.com/zboyr/lexical-fear` | `code/`, plus `data/results/`, the small final result files. These are enough for the no-GPU headline-number check below. |
| Hugging Face dataset: [`vhboyr/lexical-fear`](https://huggingface.co/datasets/vhboyr/lexical-fear) | The **full** `data/` tree (about 1 GB), laid out exactly as `data/` in this repository: datasets, judge outputs, labels, prompt pools, prompts, per-run artifacts and results. |

To get the full data, run this from the repository root:

```bash
pip install -U huggingface_hub
huggingface-cli download vhboyr/lexical-fear --repo-type dataset --local-dir data
```

This fills `data/` in place. The `data/results/` files already in the GitHub
checkout are the same files as the ones on Hugging Face, so overwriting them
changes nothing. (Newer versions of `huggingface_hub` also ship the same
command as `hf download`.)

## Repository layout

```
code/                     [GitHub]  every stage script (flat, one file per stage)
  extract_features_v2.py            hidden-state feature-cache extraction (all models)
  x58_methods.py                    the refusal2 / comply2 judge prompts and validators
  x66_*.py                          label overlays + frozen split protocol
  x71_*.py, x75_*.py                refusal-weighted probe and rank-one compiler ("path A")
  x80_*.py ... x89_*.py             one prefix per experiment (see Reproduction guide)
  x80.sbatch x81.sbatch x83.sbatch x85.sbatch x86.sbatch   Slurm wrappers for GPU stages
  lib/                              edit_common.py (model/config helpers),
                                    edit_rank1.py (the rank-one edit mechanism)
  mining/                           prompt-pool mining and delexicalization audits
                                    (s1-s12, t12-t15)
  tests/                            unit tests (run_tests.py + test_*.py)
data/
  results/                [GitHub + HF]  final aggregate result files (the paper's numbers)
  datasets/               [HF]  the frozen harmful prompt set
  judged/                 [HF]  raw judge outputs behind the Qwen labels
  labels/                 [HF]  per-model refusal/compliance label files
  pools/                  [HF]  mined prompt pools and the benign source pool
  prompts/                [HF]  the generation payload for the frozen harmful set
  runs/                   [HF]  per-experiment artifacts (cohorts, manifests, generations,
                                judgments, probes, adapters, directions, benchmark scores)
  x61_mask_tbg.npy        [HF]  retention mask over the 30,000-row feature-cache order
```

There is no `data/scores/` directory. No experiment in this release writes one.

### `code/`

- **Top-level stage scripts.** Each experiment is a set of `x<NN>_<stage>.py`
  files, with shared constants in `x<NN>_common.py`. All scripts locate
  `data/` relative to their own path, so run them from anywhere inside the
  checkout. Most write their default outputs to `data/runs/x<NN>/` and
  `data/results/`.
- **`code/lib/`.** `edit_common.py` handles model loading, the
  multimodal-wrapper fallback, chat formatting and config loading.
  `edit_rank1.py` installs the trainable rank-one `down_proj` edit used by
  path A.
- **`code/mining/`.** Builds and audits the confusable prompt pools.
  `s1` downloads the public source corpora into `pool.parquet`. `s2`/`s3`
  run the TF-IDF source-AUC screens. `s4`-`s6` do AFLite-style adversarial
  filtering. `s7`-`s9` are validity checks. `s10`-`s12` extend the pool to
  10k/15k/20k per side. `t12`-`t15` stage pools for the external labelling
  step (see [What is not included](#what-is-not-included-and-how-to-regenerate-it)).
- **`code/tests/`.** 51 unit tests in eight files, run by `run_tests.py`.
- **`*.sbatch`.** Thin Slurm wrappers that dispatch on an environment
  variable `STAGE`. See [Notes on the Slurm files](#notes-on-the-slurm-sbatch-files).

### `data/`

| Path | Contents |
|---|---|
| `data/datasets/swse_x61_majority_harmful_v1/` | The frozen harmful prompt set. `prompts.jsonl` holds the 13,527 retained harmful prompts. `excluded.jsonl` holds the 1,473 dropped prompts with each judge's code and the exclusion reason. `manifest.json` gives counts and the retention rule (three-judge majority vote). Each record carries `source_id`, the prompt id, and `source_row`, its row in the 15,000-row harmful source pool. `data/datasets/CURRENT.json` is a pointer left over from the parent project. It names a different, older dataset, and nothing in `code/` reads it. |
| `data/prompts/x63_prompts13527.json` | The generation payload: `[{id, prompt}]` for the 13,527 frozen harmful prompts. |
| `data/pools/cp_15k_benign.json` | The 15,000-row benign source pool (`source == "hard"`). The first 10,500 rows of a fixed seed-42 permutation make up the benign side of the frozen set. |
| `data/pools/mined_pool_{A_rand,A2_rand,A3_bnd,A4_bnd}.json` | The mined 5k+5k confusable pools from the delexicalization audit. A+A2+A3 is the frozen 15k scale-out; A4 is the rejected 20k extension. |
| `data/judged/x64_{refusal2,comply2}_qwen_s42.jsonl` | Per-prompt judge outputs over the ten Qwen3.5-4B responses per prompt (one boolean per response). |
| `data/labels/{tag}_labels_{v4,v5}.json` | Frozen per-model label files for the nine campaign models (`qwen`, `qwen08b`, `qwen2b`, `qwen9b`, `qwen27b`, `llama`, `gemma`, `phi`, `smollm3`). v4 stores `refusal`, `refusal_rate`, `refusal_entropy`, `complied` and `T`. v5 stores the three-class rebuild. See [Terms](#terms-used-in-this-readme-and-in-the-code). |
| `data/x61_mask_tbg.npy` | Boolean mask over the 30,000-row cache order (15,000 harmful followed by 15,000 benign). `False` on the 1,473 non-retained harmful rows. |
| `data/runs/x66/` | Frozen split index files `{tag}_seed{42,123,7}_{train,test}_indices.json` (rows of the 30,000-row cache order) for every model tag. The paper uses seed 42 only. |
| `data/runs/x80/` | Paired refusal/compliance probe artifacts. This experiment is a negative precursor (see below). |
| `data/runs/x81/` | Lexicon, fill distributions, semantic judgments, frozen arms, generations and refusal judgments for the word-removal and ignore-instruction experiments. |
| `data/runs/x83/` | Frozen keyword-insertion cohort, detection gate, generations and refusal judgments for arms O/S/M/E. |
| `data/runs/x85/` | Frozen crossed prompt-by-word cohort, plus generations and refusal judgments for arms O/H/N. The hidden-state shards are **not** included. |
| `data/runs/x86/` | Intervention manifest, fold directions, KL calibration, and refusal2/comply2 judgments for all conditions. The generations are **not** included. |
| `data/runs/x87/` | External-benchmark manifest, the Qwen-native path-A port (probes, weight audit, rank-one adapter, config, dose-KL curve), and per-condition generations plus OR-Bench, StrongREJECT, IFEval and GSM8K outputs. |
| `data/runs/x88/` | Eight-model transfer: manifest, and per-model directions, adapters, configs, generations and benchmark outputs (`data/runs/x88/<tag>/`). |
| `data/runs/x89/` | Positive-dose extension: per-condition generations and benchmark outputs. |
| `data/results/` | Final aggregate result files (one or more per experiment), listed in the [claims table](#paper-claims-figures-and-tables-result-files-and-scripts). |

Many JSON artifacts carry `inputs` / `file_record` blocks (path, size,
sha256) and a `model_snapshot` field. Paths in them are recorded relative
to the repository root. They are provenance records, not instructions. The one exception is `model_snapshot`,
which later GPU stages use to load the model; see
[Model snapshots](#model-snapshots-read-this-before-running-any-gpu-stage).

## Terms used in this README and in the code

| Term | Meaning |
|---|---|
| `refusal2` | A judge label per response: *does the response contain an explicit refusal or declination statement?* ("I can't help with that..."). It is a pure surface fact and the paper's primary label. In a pilot it was highly repeatable within and across judge families. |
| `comply2` | A judge label per response: *did the response substantively attempt to deliver what was asked*, fully or partially, whatever the framing? Content wins over disclaimers, and harmfulness does not matter. It is a different construct from `refusal2` and noisier across judges. The paper uses it only as a secondary outcome, and a refusal delta is never pooled with a compliance delta. |
| `r` (refusal rate) | The fraction of 10 sampled responses to a prompt judged `refusal2 = true`. |
| v4 / v5 targets | Two blends stored in the label files. **v4** (used here): `T = 0.7·(1 − r) + 0.3·H2(r)`, with `H2` the normalized binary entropy. The path-A probe itself is fit on `r` directly. **v5** (included for completeness): a three-class rebuild from `refusal2` and `comply2` (comply without refusal / comply with refusal / no attempt). |
| Judge | `deepseek/deepseek-v4-flash-0731` via OpenRouter, reasoning effort low, `refusal2`/`comply2` prompts verbatim from `code/x58_methods.py`, attempt seeds 42/1042/2042. The external benchmarks (x87-x89) use **no** external API. There, OR-Bench is judged by a fresh, unedited copy of Qwen3.5-4B. |
| Sampling contract | Every refusal-rate generation uses T = 1.0, top-p 0.9, top-k 50, 10 sequences, 2,048 new tokens, and one 4,096-token retry if any sequence truncates. Responses still truncated after the retry are kept. Intervention and external-benchmark runs generate greedily instead. |
| TBG / boundary tokens | "Token before generation." Qwen3.5-4B's non-thinking chat template ends with `</think>` followed by `"\n\n"`. In the code these two positions are `tbg_minus1` and `tbg`. |
| State *k* | Hugging Face `output_hidden_states` index. State 0 is the embedding layer; state *k* is the output of decoder block *k* − 1. The boundary write hooks state 30 (block 29) of Qwen3.5-4B. |
| Path A: refusal-weighted rank-one edit | A logistic probe on one cached state predicts `r`, with examples weighted by `r`. Undoing the feature standardization gives a residual-space direction `q̂`, which is compiled into a single MLP `down_proj` as `W ← W + s·q̂ vᵀ` with only `v` trained, under a 0.02 benign-KL budget. The deployment **scale** `s` is a wrapper multiplier: negative values suppress refusal. Conditions are named `x75_m3` (scale −3), `x75_p3` (+3), and so on. |
| Path B: crossed differencing, boundary write | 96 prompts crossed with 24 harm words and 24 length-matched neutral words, each word appended. The mean harm-minus-neutral hidden-state shift at the boundary tokens, computed on folds that exclude the target prompts and words, is the direction. The **boundary write** adds `dose · shift` at the two boundary positions during prefill only. Dose 1 is one observed mean shift in the model's own residual units. Conditions are named `x86_full` (dose −0.5), `x86_m150` (−1.5), `x86_p050` (+0.5), and so on. |
| Doses | Boundary-write doses and rank-one scales are in different units and are never comparable. Neither transfers numerically across models. |
| Arms | x83: **O** original, **S/M/E** word inserted at start/middle/end. x85/x86: **O** original, **H** harm word appended, **N** neutral word appended; x86 also has **B** for sealed benign prompts. x81: **A** word deleted, **B** visible mask sentinel, **C** neutral replacement; **P/D/E** plain / generic "ignore it" note / named note. |
| Pre-registered vs descriptive | Only the x87 external-evaluation gates were pre-registered. The x87 deep-dose extension, the GSM8K check, x88 and x89 are post hoc and descriptive: single seed, uncorrected intervals, `claimable: false` in their result files. |
| `x<NN>` | Experiment numbers, kept in file names for provenance. x58-x89 are the numbers used in this project. |

## Quick check: headline numbers without GPUs

### Level 1: from `data/results/` alone (GitHub checkout, no download)

This needs only Python's standard library:

```bash
python - <<'EOF'
import json
R = "data/results/"
def L(f):
    return json.load(open(R + f))

a3, a4 = L("extend_pool_A3_auc.json"), L("extend_pool_A4_auc.json")
print("15k pool TF-IDF AUC word/char:", a3["A+A2+A3 (15k+15k) word"], a3["A+A2+A3 (15k+15k) char"])
print("20k pool TF-IDF AUC word/char:", a4["A..A4 (20k+20k) word"], a4["A..A4 (20k+20k) char"])

x83 = L("x83_keyword_insertion.json")["primary_admitted_delta_vs_O"]
for pos in "SME":
    print(f"insertion {pos}: {x83[pos]['estimate']:+.3f}  CI95 {[round(v, 3) for v in x83[pos]['ci95']]}")

x85 = L("x85_crossed_keyword_latent.json")
for pos in ("tbg_minus1", "tbg"):
    g, b = x85["primary_representation_gate"][pos], x85["behavior_association_gate"][pos]
    print(f"crossed {pos}: held-out projection {g['mean_projection']:.2f}, residualised rho {b['association']['rho']:.2f}")

x86 = L("x86_boundary_intervention.json")
for k in ("suppress_H_full", "induce_N_full", "true_minus_random", "true_minus_permutation", "tbg_bidirectional_full"):
    r = x86["refusal"][k]
    print(f"intervention {k}: {r['estimate']:+.3f}  CI95 {[round(v, 3) for v in r['ci95']]}")
print("benign cost B_both_full:", round(x86["benign_over_refusal"]["delta_vs_B0"]["B_both_full"]["estimate"], 3))

x87 = L("x87_external_safety_utility.json")
for cond in ("x86_full", "x75_m3", "x75_p3"):
    c = x87["comparisons"][cond]
    print(f"external {cond}:", {e: round(c[e]["mean_delta"], 3) for e in
          ("orbench_hard_refusal", "orbench_toxic_refusal", "strongreject_score", "ifeval_prompt_strict")})
print("successful_candidates:", x87["successful_candidates"])

for tag, m in L("x88_cross_model_deep_dose.json")["models"].items():
    c = m["comparisons"].get("x86_m150", {}).get("orbench_hard_refusal")
    if c:
        print(f"transfer {tag}: write -1.5, Hard-1K {c['mean_delta']:+.3f}")
EOF
```

Expected output, abridged:

| Quantity | Value |
|---|---|
| 15k pool TF-IDF AUC (word / char) | 0.5073 / 0.6164 |
| 20k pool TF-IDF AUC (word / char) | 0.7154 / 0.7219 |
| Insertion S / M / E (refusal increase over O) | +0.305 / +0.189 / +0.326 |
| Crossed held-out projection (`tbg_minus1` / `tbg`) | 6.28 / 7.20; residualised ρ 0.65 / 0.63 |
| Boundary intervention: suppress H / induce N | +0.080 / +0.086 |
| True − random / true − permutation null | +0.088 / +0.055 |
| Final `"\n\n"` token alone | +0.070; benign over-refusal cost +0.058 |
| External, write −0.5 (Hard / Toxic / StrongREJECT / IFEval) | −0.032 / −0.011 / +0.005 / +0.018 |
| External, rank-one −3 | −0.078 / −0.017 / −0.014 / −0.004 |
| External, rank-one +3 (sign control) | +0.058 / +0.006 / +0.022 / −0.002 |
| `successful_candidates` (all three guards certified) | `[]`, a mixed guard verdict |
| Transfer, write −1.5, Hard-1K | −0.014 (llama) to −0.228 (qwen2b) on all 8 models |

### Level 2: re-run the analyses from the released per-run artifacts (HF data, CPU only)

After [downloading the data](#getting-the-data), these analysis scripts
rebuild their result files purely from files under `data/`. They need
`numpy` and `scipy`; no GPU, model or API key. Each runs in under about
10 seconds.

**Every analysis script writes into `data/results/` by default.** To compare
rather than overwrite, pass `--out` pointing somewhere else (or work in a
copy of the checkout):

```bash
mkdir -p /tmp/lf_check
python code/x83_analyze.py      --out /tmp/lf_check/x83_keyword_insertion.json
python code/x81_analyze.py      --out /tmp/lf_check/x81_lexical_trigger.json
python code/x81_analyze_de.py   --out /tmp/lf_check/x81_de_self_recovery.json
python code/x88_analyze.py      --out /tmp/lf_check/x88_cross_model_deep_dose.json
python code/x89_analyze.py      --out /tmp/lf_check/x89_positive_dose_extension.json
# x81_posthoc_fragment_sensitivity.py has no --out flag; it rewrites
# data/results/x81_posthoc_fragment_sensitivity.json in place.
```

`x89_analyze.py` also re-runs the full seeded x87 bootstrap schedule. It
**hard-fails unless every stored x87 comparison in
`data/results/x87_external_safety_utility.json` reproduces exactly**, which
makes it the quickest end-to-end check of the pre-registered external
evaluation.

`x87_analyze.py` and `x87_gsm8k_analyze.py` re-run the pre-registered
external evaluation and the post-hoc GSM8K check. The x89 conditions are the
tail of `EXTENSION_CONDITIONS` in `code/x87_common.py`, and their artifacts
live in `data/runs/x89/` under an `x89_` prefix; both analyzers find them
there (`--x89-run-dir`, default `data/runs/x89`):

```bash
python code/x87_analyze.py       --out /tmp/lf_check/x87_external_safety_utility.json
python code/x87_gsm8k_analyze.py --out /tmp/lf_check/x87_gsm8k_posthoc.json
```

Both reproduce every released x87 comparison exactly.

What to expect when diffing against the released files:

- Statistics agree exactly, apart from occasional last-digit floating-point
  differences in scipy p-values and correlations.
- `inputs` blocks differ: paths are now local, and some hashes differ.
- The released `x87_external_safety_utility.json` and
  `x89_positive_dose_extension.json` carry a hand-added `corrections` block
  recording the 2026-09-09 correction of the non-inferiority margin from a
  transcribed 0.05 to the registered 0.02. Re-runs recompute the gate
  verdicts against 0.02 (the value now in `code/x87_common.py`) but do not
  write that block.

These analyses **cannot** be re-run from released data alone:

| Script | Why |
|---|---|
| `x85_analyze.py`, `x85_aggregate_levels.py` | Need the x85 hidden-state shards (regenerable on GPU, below). |
| `x86_analyze.py` | Needs the x86 generations, which it hash-checks against the judgments (regenerable on GPU). |
| `mining/s2_auc.py`, `s3_combined_pool.py`, `s4`-`s6` | Need `data/pools/pool.parquet` (regenerable with `s1`). |
| `mining/s10`-`s12` | Need the per-round filtering parquets (`round_A_rand_{2,3,4}.parquet`), which are not released. |

## Environment setup

### Python and packages

The experiments ran on **Python 3.12**. The unit tests also pass under
Python 3.12 on CPU with older torch/transformers versions. The package list
below is derived from the imports in `code/`. Pinned versions are the ones
recorded for the GPU runs; everything else was unpinned.

```text
torch==2.11.0              # CUDA 12.8 build was used for the GPU runs
transformers==5.15.0       # several scripts rely on 5.x APIs (dtype=, get_text_config)
accelerate                 # required: models load with device_map="auto"
numpy
scipy
scikit-learn
pandas
pyarrow                    # mining scripts read/write parquet
tqdm
requests                   # OpenRouter judge client
huggingface_hub
datasets==5.0.1            # external benchmarks (x87-x89)
lm_eval==0.4.12            # IFEval / GSM8K via lm-evaluation-harness (x87-x89); the version is checked against the manifest
strong_reject              # StrongREJECT evaluator, pinned to upstream commit 7a551d5 for x87-x89
peft                       # usually needed to load the LoRA StrongREJECT evaluator
pytest                     # optional; tests fall back to a plain runner without it
llama-cpp-python           # only for mining/s8_llm_intent_judge.py
```

For example:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
python -m pip install "torch==2.11.0" "transformers==5.15.0" accelerate numpy scipy scikit-learn \
    pandas pyarrow tqdm requests huggingface_hub "datasets==5.0.1" "lm_eval==0.4.12" peft pytest
python -m pip install "git+https://github.com/dsbowen/strong_reject@7a551d5"
```

Before a long run, check the environment by **loading a model**, not just
importing packages. With transformers 5.x, `device_map="auto"` fails without
`accelerate`, and an import-only smoke test misses that. If you clone a
virtualenv, call `venv/bin/python -m pip` rather than `venv/bin/pip`: a
cloned `pip` script still points at the source environment.

### Hardware

- **CPU only:** unit tests, all Level-2 analyses, judging (API calls),
  cohort building, and the mining TF-IDF audits.
- **One GPU per job** for everything else. The runs used single NVIDIA RTX
  PRO 6000 (96 GB) or B200 (180 GB) GPUs, with jobs requesting 80 GB of
  host RAM. Qwen3.5-4B and the smaller models fit comfortably on one
  large-memory GPU. Qwen3.5-27B in bf16 needs about 54 GB for the weights
  alone and ran on a B200.
- **Disk:** about 46 GB for the nine models' feature caches, if you
  regenerate them, plus Hugging Face model caches.

### Models

All models are loaded in bf16 through Hugging Face `transformers`. Revisions
are the exact snapshots recorded in the frozen manifests.

| Tag | Hugging Face id | Revision | Used in |
|---|---|---|---|
| `qwen` | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | every experiment; also the OR-Bench judge in x87-x89 |
| `qwen08b` | `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | x88 |
| `qwen2b` | `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` | x88 |
| `qwen9b` | `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | x88 |
| `qwen27b` | `Qwen/Qwen3.5-27B` | `fc05daec18b0a78c049392ed2e771dde82bdf654` | x88 |
| `llama` | `unsloth/Llama-3.2-3B-Instruct` | `006f5dcd1393c3add266de40994ba96225e9689d` | x88; `mining/s9` |
| `gemma` | `google/gemma-4-e2b-it` | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` | x88 |
| `phi` | `microsoft/Phi-4-mini-instruct` | `cfbefacb99257ffa30c83adab238a50856ac3083` | x88 (native implementation; see `X63_TRUST_REMOTE_CODE`) |
| (evaluator) | `qylu4156/strongreject-15k-v1` (LoRA on `google/gemma-2b`) | latest at download | StrongREJECT scoring, x87-x89 |
| (embedding) | `BAAI/bge-base-en-v1.5` | latest | `mining/s7` |

`data/labels/` also holds `smollm3` labels from the parent campaign. No
experiment in this release uses that model.

**Gated repositories.** `google/gemma-2b`, the base model of the
StrongREJECT evaluator, is gated: accept its license on your Hugging Face
account and authenticate (`huggingface-cli login` or `HF_TOKEN`) before
scoring. Gemma-family models such as `google/gemma-4-e2b-it` may also
require accepting the Gemma terms. Some source corpora downloaded by
`mining/s1_download_extract.py` may require accepting dataset terms on
Hugging Face.

**Offline compute nodes.** If your GPU nodes have no internet access, cache
every model and dataset on a login node first and run jobs with
`HF_HUB_OFFLINE=1`, as the `.sbatch` files do. Besides the models, that
means `bench-llm/or-bench` (revision `e36d8b80e81837c8a8f264bbb2a49f1b32c7e272`,
configs `or-bench-hard-1k` and `or-bench-toxic`), `openai/gsm8k`, the
datasets behind lm-eval's `ifeval` task, and, for `x87_build_manifest.py`,
the StrongREJECT CSV at upstream commit `f7cad6c17e624e21d8df2278e918ae1dddb4cb56`
(fetched from GitHub when the manifest is built). Once built, the manifest
stores the benchmark rows.

### Model snapshots (read this before running any GPU stage)

Most GPU stages do **not** take a `--model` argument. They load the model
from the model reference recorded in a frozen design artifact,
so the checkpoint cannot drift:

| Stage scripts | Read the snapshot from |
|---|---|
| `x81_score_fills`, `x81_judge_semantics`, `x81_generate`, `x81_build_lexicon` | `data/runs/x81/x81_audit.json` → `model_snapshot.snapshot` |
| `x83_detect`, `x83_generate` | `data/runs/x83/x83_cohort.json` → `model_snapshot` |
| `x85_extract_hidden`, `x85_generate` | `data/runs/x85/x85_cohort.json` → `model_snapshot` |
| `x86_build_manifest`, `x86_calibrate_kl`, `x86_generate` | x85 cohort / `data/runs/x86/x86_manifest.json` → `model_snapshot` |
| `x87_generate`, `x87_score`, `x87_ifeval`, `x87_gsm8k` | `data/runs/x87/x87_manifest.json` → `model_snapshot.snapshot` |
| `x88_*` | `data/runs/x88/x88_manifest.json` → `models.<tag>.snapshot`, and `judge_snapshot.snapshot` for the OR-Bench judge |

Each reference is written as `<org>/<name>@<revision>`, for example
`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`: the Hugging Face
repo id plus the exact commit that was used. The scripts pass this string
straight to `from_pretrained`, which loads it as a local directory if one of
that name exists. You have two options:

1. **Reuse the frozen designs (recommended).** From the repository root, run

   ```bash
   python code/link_model_snapshots.py                       # every model referenced in data/
   python code/link_model_snapshots.py --only Qwen/Qwen3.5-4B  # or just the ones you need
   ```

   It downloads each recorded revision into your Hugging Face cache and
   creates a symlink `<org>/<name>@<revision>` pointing at the cached
   snapshot, both in the repository root and in `code/` (the sbatch wrappers
   run from `code/`). If you run a stage from another directory, pass
   `--dest <that directory>`. Gated models need `HF_TOKEN` and an accepted
   license. Do not edit the JSON: downstream stages and analyzers check
   recorded sha256 values.
2. **Re-freeze from the start of an experiment** by passing your own
   snapshot path to the stage-0 builder (`--model-snapshot` on
   `x81_audit.py`, `x83_build_cohort.py`, `x85_build_cohort.py` and
   `x87_build_manifest.py`). `x88_build_manifest.py` resolves snapshots
   with `huggingface_hub.snapshot_download`. Re-freezing changes the
   manifest hashes, so the result is a fresh, independent run rather than
   a byte-level reproduction.

Other path fields in the released artifacts (`path`, `inputs`, `out`, ...)
are recorded relative to the repository root.

### Environment variables and secrets

| Variable | Read by | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` (in a **`.env` file**, not the environment) | `x81_judge_refusal.py`, reused by `x83_judge_refusal.py`, `x85_judge_refusal.py`, `x86_judge.py` | The LLM judge. The key is read from a file named `.env` **at the repository root** (next to `code/`), from a line `OPENROUTER_API_KEY=<your key>`. Shell environment variables are ignored. Keep `.env` out of version control. |
| `SWSE_FC` | `x66_build_v4v5_overlays.py` | Root directory of the feature caches. `X66_FC` is an older alias. If unset, it falls back to `feature_caches/` at the repository root. Other scripts take cache paths as explicit arguments; the commands below write those paths as `$SWSE_FC/...` for consistency. |
| `X66_DATA`, `X66_CLUSTERED` | `x66_build_v4v5_overlays.py` | Override the `data/` directory and the harmful clustered pool path (default `data/pools/cp_15k_clustered_v2.json`). |
| `X63_TRUST_REMOTE_CODE` | `extract_features_v2.py`, `lib/edit_common.py` (set automatically by `x88_common.py`) | `0` forces the native `transformers` implementation instead of repository custom code. Needed for `microsoft/Phi-4-mini-instruct`, whose bundled `modeling_phi3.py` does not work with transformers 5.x. The default is `1`. |
| `HF_HOME`, `HF_HUB_OFFLINE`, `HF_TOKEN` | Hugging Face libraries (`x75_preflight.py` also checks `HF_HUB_OFFLINE`) | Cache location, offline mode, and gated-model access. |
| `STAGE`, `CONDITION`, `SEED`, `EXT`, `SHARD`, `NSHARDS` | the `.sbatch` wrappers only | Select which stage a Slurm job runs. |

The judge targets `deepseek/deepseek-v4-flash-0731` through OpenRouter's
chat-completions endpoint. Judge spend was small: about $0.10 per condition
for the 10-responses-per-call `refusal2` contract, and under about $20 for
an entire experiment. `x81_judge_refusal.py` defaults to `--threads 200`;
the later judge scripts default to 24. Lower the thread count if you hit
rate limits. Judge scripts resume: re-running appends only the rows that
are still missing or failed.

## What is not included, and how to regenerate it

| Missing artifact | Size | Needed by | How to regenerate |
|---|---|---|---|
| Feature caches `$SWSE_FC/{tag}_A15k_v2(style)_full` and the overlays `{tag}_A15k_{v4,v5}_full` | ~46 GB for nine models | path A: `x71_prepare_dataset.py` in the x87 and x88 stage 0, and `x80_fit_paired_probes.py` | `code/extract_features_v2.py`, then `code/x66_build_v4v5_overlays.py` ([commands](#stage-2-feature-caches-labels-overlays-and-the-frozen-split-x66)). Both read `data/pools/cp_15k_clustered_v2.json` (on Hugging Face). |
| `data/pools/pool.parquet`, the raw mined candidate pool | 100 MB | `mining/s2`-`s6` | `python code/mining/s1_download_extract.py` (downloads the public source corpora). The source dataset revisions are not pinned, so the rebuilt pool may differ from the original. |
| `data/pools/round_A_rand_{2,3,4}.parquet` and other per-round filtering outputs | n/a | `mining/s10`-`s12` | Written by `mining/s6_iterative_random_final.py` from `pool.parquet`. An exact match is not guaranteed if `pool.parquet` differs. |
| Clustered staging batches (`cp_clustered.json`, `cp_rest_clustered.json`, `cp_a2_clustered.json`, `cp_a3_clustered.json`) | n/a | `mining/t13`, `t15` | Not included. They came from the same external labelling step. |
| `data/generations/`, the raw 10-per-prompt response sets behind the labels | 3.1 GB | none of the released scripts; the labels are released | Not included. Generating them needs the sampling contract above and the generation script from the parent project, which is not part of this release. |
| `data/runs/x85/features/x85_hidden_shard_{000..023}-of-024.pt`, residual states for 4,704 cells × 4 positions × 33 states, fp16 | 3.0 GB | `x85_analyze.py`, `x85_aggregate_levels.py`, `x86_build_manifest.py`, and the paper's projection figure | `x85_extract_hidden.py --shard i --num-shards 24` for i = 0..23 ([x85](#x85-crossed-prompt-by-word-decomposition)) |
| `data/runs/x86/x86_generations_<condition>.json`, 19 files | ~140 MB | `x86_analyze.py` | `x86_generate.py --condition <c>` from the released manifest, KL calibration and directions ([x86](#x86-held-out-boundary-intervention)) |
| `data/runs/x87/x75_qwen_prepared/` (`probe_dataset.pt`, `prompt_splits.json`) and the per-model prepared tensors for x88 | 2.2 GB for Qwen | re-running path-A stage 0 | `x71_prepare_dataset.py` (needs a feature cache) |
| Full lm-eval GSM8K logs | n/a | nothing | The released `x87_gsm8k_*.json` files keep per-question scores but omit the `doc`/`arguments` fields. |
| x75 Llama-era artifacts (`data/runs/x71/`, `data/runs/x75/`) | n/a | `x75_freeze_config.py`, `x75_preflight.py`, `x75_dose_kl.py`, `x75_analyze.py`, `x71_analyze.py` | Not included. These scripts document the Llama experiment that produced the path-A method; the paper reports only its Qwen-native port, which is fully reproducible ([x87 stage 0](#x87-external-benchmarks-pre-registered)). |
| Producer scripts for `data/runs/x80/x80_labels.npz`, `data/results/x80_probe_geometry.json`, `data/results/confusable_A_auc.json` and `data/runs/x87/x87_x75_qwen_dose_kl.json` | n/a | n/a | Not included in `code/`. The artifacts themselves are released. |
| Plotting scripts for the paper figures | n/a | n/a | Not part of `code/`. The [claims table](#paper-claims-figures-and-tables-result-files-and-scripts) lists the data each figure is drawn from. |

## Reproduction guide

The pipeline runs in this order: **mining and dataset freeze → labels →
feature caches, overlays and splits → experiments.** Each frozen artifact
is released, so you can start at any stage. Unless noted, run commands from
the repository root. Each experiment section gives a one-line purpose, the
commands in order, inputs, outputs and approximate compute. "GPU" means one
GPU; "API" means OpenRouter judge calls.

Timing figures come from the original runs: HF `generate` handles roughly
215-255 sampled rows per GPU-hour under the 10-sample contract and about 40
greedy benchmark rows per GPU-minute.

### Stage 1: mining the confusable pools (delexicalization audit)

*Purpose:* build harmful/benign prompt pools that a TF-IDF classifier cannot
separate by source, and show that lexical separability keeps resurfacing.

```bash
python code/mining/s1_download_extract.py          # -> data/pools/pool.parquet (downloads public corpora)
python code/mining/s2_auc.py                       # -> data/results/auc_pairings.json
python code/mining/s3_combined_pool.py             # -> data/results/combined_pool_auc.json, data/pools/combined_pool.json
python code/mining/s4_adversarial_filter.py        # single-round filter -> data/results/adversarial_filter_results.json
python code/mining/s5_iterative_filter.py          # iterative, hardness-tail cut -> data/results/iterative_filter_results.json
python code/mining/s6_iterative_random_final.py    # iterative, random final cut -> data/pools/mined_pool_A_rand.json (+ round parquets)
python code/mining/s10_extend_pool_A2.py           # -> data/pools/mined_pool_A2_rand.json, data/results/extend_pool_A2_auc.json
python code/mining/s11_extend_pool_A3.py           # -> data/pools/mined_pool_A3_bnd.json, data/results/extend_pool_A3_auc.json  (15k: frozen)
python code/mining/s12_extend_pool_A4.py           # -> data/pools/mined_pool_A4_bnd.json, data/results/extend_pool_A4_auc.json  (20k: rejected)
# validity checks on pool A (outputs not part of the release):
python code/mining/s7_semantic_separability.py     # BAAI/bge-base-en-v1.5 embedding separability (GPU optional)
python code/mining/s8_llm_intent_judge.py [--pool P --tag T --n N]   # local GGUF judge; needs s2_cluster_local.py (not included)
python code/mining/s9_llama_hb_classify.py [--pool P --tag T --n N --seed S | --all]   # Llama-3.2-3B zero-shot H/B (GPU)
# staging for the external labelling step:
python code/mining/t12_prepare_a2.py; python code/mining/t14_prepare_a3.py
python code/mining/t13_prepare_full10k.py; python code/mining/t15_prepare_full15k.py   # need clustered batches (not included)
```

- *Inputs:* public datasets (`heegyu/wildjailbreak-train`, `bench-llm/or-bench`,
  `AmazonScience/FalseReject`, `furonghuang-lab/PHTest`, `allenai/coconot`,
  `natolambert/xstest-v2-copy`, `JailbreakBench/JBB-Behaviors`,
  `lmsys/toxic-chat`, `OpenSafetyLab/Salad-Data`, `Babelscape/ALERT`).
- *Outputs:* the released ones are `data/pools/mined_pool_*.json` and
  `data/results/{auc_pairings,combined_pool_auc,extend_pool_A{2,3,4}_auc}.json`.
- *Compute:* CPU, minutes per step; `s1` is dominated by downloads.
- *Caveats:* `s10`-`s12` need the round parquets written by `s6`.
  `s8_llm_intent_judge.py` imports `s2_cluster_local.py` and a local GGUF
  model, neither of which is included. The in-loop filtering trajectory
  quoted in the paper (0.98 → near chance, and the 0.70 fresh-auditor
  rebound) comes from the original run log; its per-round result files are
  not part of this release.

### Dataset freeze (released, no script)

The frozen set has 13,527 harmful prompts and 10,500 benign prompts:

- **Harmful side.** The 15,000 harmful rows of the clustered pool, minus
  1,473 rows dropped by a three-judge majority vote (`manifest.json` holds
  the rule). The retention builder belongs to the parent project.
  `data/x61_mask_tbg.npy` enforces the result in the feature-cache order.
- **Benign side.** The first 10,500 rows of `np.random.RandomState(42).permutation`
  over the `source == "hard"` rows of `cp_15k_benign.json`, applied by
  `x66_split_cache_masked.py --benign_n 10500`.

**Positional contract:** every cache, mask and index file uses the same
30,000-row order. Harmful rows come first, sorted by id, then benign rows,
sorted by id.

### Labels: `refusal2` / `comply2` (released)

Each of the 13,527 harmful prompts was answered 10 times by each model under
the sampling contract and judged once per method (`refusal2` and `comply2`,
judge seed 42). The Qwen3.5-4B judge outputs are in `data/judged/`. The
frozen per-model label files are in `data/labels/`. Downstream code only
reads these files; the label-building script and the raw responses belong
to the parent project and are not included. `code/x58_methods.py` holds the
exact judge prompts and validators reused by every judge script here.

### Stage 2: feature caches, label overlays and the frozen split (x66)

*Purpose:* build the 30,000-row hidden-state caches that path A reads, bake
the retention mask and v4/v5 labels into them, and (optionally) regenerate
the frozen split indices.

```bash
export SWSE_FC=/path/to/feature_caches
# 1) one forward pass per prompt; caches TBG and content-mean states for every layer, plus TBG logits.
#    Directory names must match SRC_CACHE in code/x66_build_v4v5_overlays.py,
#    e.g. qwen35_4b_A15k_v2_full for qwen, llama32_A15k_v2_full for llama, <tag>_A15k_v2style_full otherwise.
python code/extract_features_v2.py --model Qwen/Qwen3.5-4B \
    --cache_dir "$SWSE_FC/qwen35_4b_A15k_v2_full" \
    --clustered data/pools/cp_15k_clustered_v2.json \
    --benign data/pools/cp_15k_benign.json [--device cuda]
# (Phi: prefix with X63_TRUST_REMOTE_CODE=0)

# 2) v4/v5 label overlays: {tag}_A15k_v4_full and {tag}_A15k_v5_full (symlinked features + real label arrays + mask)
python code/x66_build_v4v5_overlays.py --tag qwen     # tags: qwen qwen08b qwen2b qwen9b qwen27b llama gemma phi smollm3

# 3) optional: regenerate the frozen seed-42 split (the released index files are what everything uses)
python code/x66_split_cache_masked.py --src "$SWSE_FC/qwen_A15k_v4_full" \
    --train /path/to/split/train --test /path/to/split/test --seed 42 --benign_n 10500
#    writes sliced caches plus split_indices.json in each output dir; the released
#    data/runs/x66/{tag}_seed42_{train,test}_indices.json files are these indices.
```

- *Inputs:* `cp_15k_clustered_v2.json` (the harmful source pool; on Hugging Face),
  `cp_15k_benign.json`, and `data/labels/{tag}_labels_{v4,v5}.json`.
- *Outputs:* `$SWSE_FC/<cache>/{hidden_states.pt, tbg_logits.npy, prompts.json, *_labels_tbg.npy}`
  and the overlay directories.
- *Compute:* one GPU forward pass per prompt (30,000 prompts per model). The
  state arrays are held in host RAM before saving: 2 × layers × 30,000 ×
  hidden size × 2 bytes, about 10 GB for Qwen3.5-4B. About 46 GB on disk for
  all nine models.

### x80: paired refusal and compliance probes (negative precursor)

*Purpose:* test whether refusal (`refusal2`) and substantive compliance
(`comply2`) give separable probe directions worth editing. The
pre-registered gate G1 **fails**: the two labels are near-complementary
(only 4.24% of responses disagree), so the edit stages never ran. Do not
cite `x80_*` numbers as evidence for the paper's claims.

```bash
python code/x80_label_audit.py        # CPU; writes data/runs/x80/x80_label_manifest.json, data/results/x80_label_audit.json
python code/x80_fit_paired_probes.py --cache "$SWSE_FC/qwen_A15k_v4_full/hidden_states.pt" \
    --labels data/runs/x80/x80_labels.npz --indices-dir data/runs/x66 \
    --seed 42 --out-dir data/runs/x80/probes --n-perm 5 --n-boot 50 [--l2-ext]
# repeat for --seed 123 and --seed 7; --l2-ext is the extended-L2-grid sensitivity run (data/runs/x80/probes_l2ext)
```

- *Compute:* one GPU, up to 3 h per seed (the `x80.sbatch` limit).

### x81: removing a refusal-associated word, and "ignore it" instructions

*Purpose:* when the model itself judges that swapping a discovery-fitted
refusal-associated word keeps the prompt's meaning, does refusal fall?
Result: **inconclusive** (Δ = +0.013, 95% CI [−0.019, +0.046]). The visible
mask sentinel raises refusal by +0.084, an artifact of the sentinel. In the
follow-up, telling the model to ignore an inserted word *raises* refusal.
This is supporting context, not a headline result.

```bash
python code/x81_audit.py --model-snapshot /path/to/Qwen3.5-4B/snapshot   # CPU (tokenizer); needs the clustered pool
python code/x81_build_lexicon.py                     # CPU; discovery (x66 seed-42 train) TF-IDF + L1 logistic
python code/x81_score_fills.py                       # GPU; [--batch-size 8]
python code/x81_judge_semantics.py                   # GPU; same-model SAME/CHANGED judgments [--batch-size 16]
python code/x81x_top30_explore.py                    # GPU; outcome-blind exploration behind the admission-rule revision
python code/x81_build_arms.py                        # CPU; freezes data/runs/x81/x81_arms.json (108 prompts)
for c in A B C; do python code/x81_generate.py --condition $c; done            # GPU
for c in A B C; do python code/x81_judge_refusal.py --condition $c --threads 24; done   # API
python code/x81_analyze.py                           # -> data/results/x81_lexical_trigger.json
python code/x81_posthoc_fragment_sensitivity.py      # -> data/results/x81_posthoc_fragment_sensitivity.json
# follow-up: instruction-based self-recovery arms P/D/E
python code/x81_build_arms_de.py                     # -> data/runs/x81/x81_arms_de.json
for c in P D E; do python code/x81_generate.py --condition $c --arms data/runs/x81/x81_arms_de.json; done   # GPU
for c in P D E; do python code/x81_judge_refusal.py --condition $c --threads 24; done                       # API
python code/x81_analyze_de.py                        # -> data/results/x81_de_self_recovery.json
```

- *Inputs:* `data/prompts/x63_prompts13527.json`, `data/labels/qwen_labels_v4.json`,
  `data/judged/x64_refusal2_qwen_s42.jsonl`, and `data/runs/x66/qwen_seed42_*`.
  Stages 0-1 also read `cp_15k_clustered_v2.json`. From stage 2 on,
  everything runs from released frozen artifacts.
- *Compute:* about 14-18 GPU-minutes per generation arm (108 prompts × 10
  responses, one B200), and a few cents of judge spend per arm.

### x83: one inserted harm keyword (the phenomenon)

*Purpose:* insert one external harm-topic word at the start, middle or end
of a prompt Qwen3.5-4B mostly answers. Refusal rises by +0.31 / +0.19 /
+0.33 over a contemporaneously regenerated original, including on prompts
where the model names the inserted word with more than 90% confidence.

```bash
python code/x83_build_cohort.py --model-snapshot /path/to/Qwen3.5-4B/snapshot   # CPU; freezes data/runs/x83/x83_cohort.json
                                                                                   # (needs the clustered pool; the frozen cohort is released)
python code/x83_detect.py                            # GPU; same-model detection gate -> data/runs/x83/x83_detect.json
for c in O S M E; do python code/x83_generate.py --condition $c; done           # GPU
for c in O S M E; do python code/x83_judge_refusal.py --condition $c; done      # API
python code/x83_analyze.py                           # -> data/results/x83_keyword_insertion.json
```

- *Inputs:* the frozen cohort (321 prompts from the x66 seed-42 evaluation
  split with historical refusal rate ≤ 0.8; the 99-word lexicon is in
  `code/x83_common.py`) and `data/labels/qwen_labels_v4.json`.
- *Outputs:* `data/runs/x83/x83_generations_{O,S,M,E}.json`,
  `x83_refusal_{O,S,M,E}.jsonl` and the result file.
- *Compute:* detection about 4 GPU-minutes; generation 63-87 GPU-minutes per
  arm; judging about $0.50 in total.

### x85: crossed prompt-by-word decomposition

*Purpose:* cross 96 prompts with 24 harm words and their tokenizer-length-matched
neutral twins, and show that at the boundary tokens the neutral-corrected
shift has a shared direction that generalizes to unseen prompts *and*
unseen words and tracks the induced refusal change. This result is
**observational**; x86 provides the causal test.

```bash
python code/x85_build_cohort.py [--model-snapshot /path/to/snapshot]   # CPU; freezes data/runs/x85/x85_cohort.json from the x83 cohort
for i in $(seq 0 23); do python code/x85_extract_hidden.py --shard $i --num-shards 24; done   # GPU
                                                     # -> data/runs/x85/features/x85_hidden_shard_{000..023}-of-024.pt
for c in O H N; do python code/x85_generate.py --condition $c; done             # GPU
for c in O H N; do python code/x85_judge_refusal.py --condition $c; done        # API
python code/x85_analyze.py                           # primary band: states 29-31, positions tbg_minus1,tbg
     # -> data/results/x85_crossed_keyword_latent.json, x85_crossed_keyword_directions.npz
python code/x85_analyze.py --states 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32 \
    --positions instruction_last,user_end,tbg_minus1,tbg \
    --out data/results/x85_crossed_keyword_latent_allstates.json \
    --directions-out data/results/x85_crossed_keyword_directions_allstates.npz
python code/x85_analyze.py --states 0 --positions instruction_last --bootstrap 10 --randomization 10 \
    --out data/results/x85_crossed_keyword_latent_state0_instruction.json \
    --directions-out data/results/x85_crossed_keyword_directions_state0_instruction.npz
python code/x85_aggregate_levels.py                  # descriptive supplement -> data/results/x85_crossed_keyword_levels.json
```

State 0 is analysed separately because at the post-word positions the
embedding-layer difference is exactly zero, which breaks the direction
estimate.

- *Compute:* extraction about 75 s per shard after a 40 s model load (about
  15 minutes for the 24-task array). Generation: arm H 2 h 16 min, N 2 h 39 min,
  O 30 min (576 / 576 / 96 rows × 10 responses). Judging about $0.55.
  Analysis a few seconds once the shards exist.

### x86: held-out boundary intervention

*Purpose:* causal test. Subtract the fold-specific boundary shift from
harm-word cells and add it to neutral-word cells, on cells whose prompt
**and** word were both held out of that fold's direction. Measure
refusal (`refusal2`, primary) and substantive compliance (`comply2`,
secondary) against KL-matched random and sign-permutation null directions.

```bash
python code/x86_build_manifest.py   # GPU (dose-zero hook audit); reads the x85 cohort + 24 shards
                                    # -> data/runs/x86/x86_manifest.json, x86_directions.npz
python code/x86_calibrate_kl.py     # GPU; benign-KL dose admission (a* = 0.5) -> data/runs/x86/x86_kl_calibration.json
# 19 conditions: 14 target + 5 sealed-benign
for c in H_0 H_both_half H_both_full H_minus1_full H_tbg_full H_random_full H_permutation_full \
         N_0 N_both_half N_both_full N_minus1_full N_tbg_full N_random_full N_permutation_full \
         B_0 B_both_half B_both_full B_random_full B_permutation_full; do
  python code/x86_generate.py --condition $c                       # GPU -> data/runs/x86/x86_generations_$c.json
  python code/x86_judge.py --condition $c --method refusal2        # API
done
for c in H_0 H_both_half H_both_full H_minus1_full H_tbg_full H_random_full H_permutation_full \
         N_0 N_both_half N_both_full N_minus1_full N_tbg_full N_random_full N_permutation_full; do
  python code/x86_judge.py --condition $c --method comply2 --per-response     # API, target conditions only
done
python code/x86_analyze.py --allow-code-drift    # -> data/results/x86_boundary_intervention.json
```

`--allow-code-drift` is required. `x86_judge.py` and `x86_analyze.py` were
changed after the manifest was frozen: a `--judge-model` override, the
`--per-response` mode, and a comply2-optional guard. The analyzer records
that drift. It still refuses to run if any frozen-pipeline file
(`x86_common`, `x86_build_manifest`, `x86_calibrate_kl`, `x86_generate`)
has changed. `comply2` must be judged with `--per-response`, one response
per API call: in the original run, 10-per-call judging failed on 40-50% of
long-response rows.

To re-run only the analysis from the released judgments, regenerate the
generations first, because the analyzer checks response hashes.

- *Compute:* stage 0 plus a 3 GB shard reload; KL calibration about 12 GPU-minutes.
  Generation of 288 cells × 10 responses per target condition (256 × 5 per
  benign condition) runs at about 230 rows/GPU-hour, so roughly 1-1.5
  GPU-hours per condition and about 25 GPU-hours in total. Judging: about
  $0.10 per condition for `refusal2`, about $4 in total for `comply2`.

### x87: external benchmarks (pre-registered)

*Purpose:* evaluate the boundary write (`x86_full`, dose −0.5) and a
Qwen-native port of the rank-one edit (`x75_m3`, scale −3; signed control
`x75_p3`, +3) on OR-Bench Hard-1K (over-refusal; superiority gate),
OR-Bench Toxic, StrongREJECT and IFEval (non-inferiority guards with a 0.02
absolute margin). Both candidates pass superiority; the guard verdict is
mixed (`successful_candidates: []`). The deep-dose extension
(`x86_m075`…`x86_m300`, `x75_m4`…`x75_m7`) and GSM8K are post hoc and
descriptive.

**Stage 0: the Qwen-native path-A port** (x71/x75 chain; needs the
`qwen_A15k_v4_full` cache):

```bash
SNAP=/path/to/Qwen3.5-4B/snapshot          # revision 851bf6e8...
CFG=data/runs/x87/x87_x75_qwen_config.json
PREP=data/runs/x87/x75_qwen_prepared
python code/x71_prepare_dataset.py \
    --hidden-states "$SWSE_FC/qwen_A15k_v4_full/hidden_states.pt" \
    --prompts "$SWSE_FC/qwen_A15k_v4_full/prompts.json" \
    --source-labels "$SWSE_FC/qwen_A15k_v4_full/source_labels_tbg.npy" \
    --clustered-order data/pools/cp_15k_clustered_v2.json \
    --refusal-labels data/labels/qwen_labels_v4.json \
    --train-indices data/runs/x66/qwen_seed42_train_indices.json \
    --test-indices data/runs/x66/qwen_seed42_test_indices.json \
    --out-dir "$PREP" --config "$CFG"
python code/x75_prepare_weighted_probe.py --dataset "$PREP/probe_dataset.pt" \
    --prompt-splits "$PREP/prompt_splits.json" --out data/runs/x87/x75_qwen_weight_audit.json
python code/x75_train_weighted_probe.py --arm refusal_weighted --dataset "$PREP/probe_dataset.pt" \
    --weight-audit data/runs/x87/x75_qwen_weight_audit.json \
    --out-dir data/runs/x87/x75_qwen_probes --config "$CFG"
python code/x75_train_edit.py --probe data/runs/x87/x75_qwen_probes/x75_refusal_weighted_true.pt \
    --prompt-splits "$PREP/prompt_splits.json" \
    --out data/runs/x87/x75_qwen_refusal_weighted_true.pt --config "$CFG" --model-source "$SNAP"
```

Always pass `--config`. Without it, `lib/edit_common.load_config` falls back
to `data/runs/x71/x71_config.json`, which is not released. The released
adapter selects state 31 and trains only `model.layers.30.mlp.down_proj.right`
(step 200, benign KL 0.0059). Stage 0 took about 5 GPU-minutes with the
cache resident.

**Stages 1-4: benchmarks.**

```bash
python code/x87_build_manifest.py --model-snapshot "$SNAP"   # -> data/runs/x87/x87_manifest.json (snapshot dir name must match x86's)
for c in base x86_full x75_m3 x75_p3 \
         x86_m075 x86_m100 x75_m4 x86_m125 x86_m150 x86_m200 x86_m300 x75_m5 x75_m6 x75_m7; do
  python code/x87_generate.py --condition $c            # GPU, greedy, <=1,024 new tokens -> x87_generations_$c.json
  python code/x87_ifeval.py   --condition $c            # GPU, lm-eval ifeval              -> x87_ifeval_$c.json
  python code/x87_score.py orbench      --condition $c  # GPU, unedited-Qwen 3-way judge  -> x87_orbench_$c.json
  python code/x87_score.py strongreject --condition $c  # GPU, official finetuned evaluator -> x87_strongreject_$c.json
  python code/x87_gsm8k.py    --condition $c            # GPU, post hoc                    -> x87_gsm8k_$c.json
done
python code/x87_analyze.py          # -> data/results/x87_external_safety_utility.json (sole gate authority)
python code/x87_gsm8k_analyze.py    # -> data/results/x87_gsm8k_posthoc.json (claimable: false)
```

`x87_analyze.py` analyzes the post-hoc dose extension only when artifacts
for **every** condition in `EXTENSION_CONDITIONS` exist. That list includes
the seven x89 conditions, whose artifacts it reads from `data/runs/x89/`
(see [x89](#x89-positive-dose-and-shallow-scale-extension-descriptive)).

- *Inputs:* `data/runs/x86/x86_directions.npz`, `x86_kl_calibration.json`,
  and the stage-0 adapter (defaults of `--x86-directions`,
  `--x86-calibration`, `--x75-adapter`).
- *Compute:* generation plus IFEval about 60-70 GPU-minutes per condition;
  OR-Bench judging 7-9 GPU-minutes; StrongREJECT about 40 s; GSM8K about
  12 GPU-minutes per condition. No external API.

### x88: eight-model transfer (descriptive)

*Purpose:* apply the boundary write at one fixed dose (−1.5) and the
rank-one edit at fixed scales (−6, plus −3 for the safety endpoints only)
to seven more models, using each model's own direction and adapter, and
compare with the imported Qwen3.5-4B row. Single seed, uncorrected intervals,
one fixed OR-Bench judge (the unedited Qwen3.5-4B), `claimable: false`.

```bash
python code/x88_make_configs.py          # -> data/runs/x88/<tag>/x88_x75_config.json   [--date "25 Aug 2026"]
python code/x88_build_manifest.py        # resolves snapshots via snapshot_download, audits boundary token ids
                                         # -> data/runs/x88/x88_manifest.json
for tag in qwen08b qwen2b qwen9b qwen27b llama gemma phi; do
  D=data/runs/x88/$tag; SNAP_T=$(python -c "import json;print(json.load(open('data/runs/x88/x88_manifest.json'))['models']['$tag']['snapshot'])")
  # (for phi, export X63_TRUST_REMOTE_CODE=0 for the x71/x75 steps)
  # stage 0: per-model path-A chain on the model's own cache, v4 labels and seed-42 split
  python code/x71_prepare_dataset.py --hidden-states "$SWSE_FC/${tag}_A15k_v4_full/hidden_states.pt" \
      --prompts "$SWSE_FC/${tag}_A15k_v4_full/prompts.json" \
      --source-labels "$SWSE_FC/${tag}_A15k_v4_full/source_labels_tbg.npy" \
      --clustered-order data/pools/cp_15k_clustered_v2.json --refusal-labels data/labels/${tag}_labels_v4.json \
      --train-indices data/runs/x66/${tag}_seed42_train_indices.json \
      --test-indices data/runs/x66/${tag}_seed42_test_indices.json \
      --out-dir $D/x75_prepared --config $D/x88_x75_config.json
  python code/x75_prepare_weighted_probe.py --dataset $D/x75_prepared/probe_dataset.pt \
      --prompt-splits $D/x75_prepared/prompt_splits.json --out $D/x75_weight_audit.json
  python code/x75_train_weighted_probe.py --arm refusal_weighted --dataset $D/x75_prepared/probe_dataset.pt \
      --weight-audit $D/x75_weight_audit.json --out-dir $D/x75_probes --config $D/x88_x75_config.json
  python code/x75_train_edit.py --probe $D/x75_probes/x75_refusal_weighted_true.pt \
      --prompt-splits $D/x75_prepared/prompt_splits.json \
      --out $D/x75_${tag}_refusal_weighted_true.pt --config $D/x88_x75_config.json --model-source "$SNAP_T"
  # per-model boundary directions from the frozen x85 cohort texts
  python code/x88_extract_directions.py --model $tag      # -> $D/x88_directions.npz, x88_directions_audit.json
  for c in base x86_m150 x75_m6 x75_m3; do
    python code/x88_generate.py --model $tag --condition $c
    python code/x88_score.py orbench      --model $tag --condition $c
    python code/x88_score.py strongreject --model $tag --condition $c
  done
  for c in base x86_m150 x75_m6; do                      # x75_m3 is safety-endpoints only
    python code/x88_lmeval.py ifeval --model $tag --condition $c
    python code/x88_lmeval.py gsm8k  --model $tag --condition $c
  done
done
python code/x88_analyze.py               # imports the qwen row from x87 -> data/results/x88_cross_model_deep_dose.json
```

The `x75_prepared` and `x75_probes` directory names above are illustrative;
only the adapter and direction paths are fixed by `x88_common.py`. The hook
block follows a frozen relative-depth rule,
`round(29/33 · n_blocks)`, with no per-model layer search.

- *Compute:* per model, stage 0 plus directions take minutes to tens of
  minutes. Generation runs 32-63 GPU-minutes per condition for the small
  models and about 2 h 20 min for `qwen27b` (on a 180 GB GPU).

### x89: positive-dose and shallow-scale extension (descriptive)

*Purpose:* run positive boundary-write doses (+0.5/+1.0/+1.5) and shallow
rank-one scales (−1, −2, +1, +2) through the unchanged x87 pipeline. The
x87 base generations are reused.

```bash
for c in x86_p050 x86_p100 x86_p150 x75_m1 x75_m2 x75_p1 x75_p2; do
  R=data/runs/x89
  python code/x87_generate.py --condition $c --out $R/x89_generations_$c.json
  python code/x87_ifeval.py   --condition $c --out $R/x89_ifeval_$c.json
  python code/x87_score.py orbench      --condition $c --generations $R/x89_generations_$c.json --out $R/x89_orbench_$c.json
  python code/x87_score.py strongreject --condition $c --generations $R/x89_generations_$c.json --out $R/x89_strongreject_$c.json
  python code/x87_gsm8k.py    --condition $c --out $R/x89_gsm8k_$c.json
done
python code/x89_analyze.py     # verifies every x87 comparison reproduces, then -> data/results/x89_positive_dose_extension.json
                               # [--skip-gsm8k]
```

- *Compute:* about 65-75 GPU-minutes per condition for generation plus
  IFEval, about 10 minutes for scoring, and GSM8K in parallel.

### x66 / x71 / x75 as prerequisites

- **x66** (overlays, masked split) is a prerequisite of every path-A step
  and of x80, through the feature caches. Its split index files are
  released, and every cohort in this paper is drawn from the seed-42
  evaluation split (1,353 harmful prompts).
- **x71** supplies the compiler: `x71_prepare_dataset.py` and
  `x71_train_edit.py`. `x75_train_edit.py` is a shim over the latter, and
  the mechanism lives in `lib/edit_rank1.py`. `x71_analyze.py` belongs to
  the Llama-era experiment and is not needed.
- **x75** supplies the refusal-weighted probe (`x75_weighting.py`,
  `x75_prepare_weighted_probe.py`, `x75_train_weighted_probe.py`). The
  stand-alone Llama x75 experiment (`x75_freeze_config.py`,
  `x75_preflight.py`, `x75_dose_kl.py`, `x75_analyze.py`) is method
  provenance only, and its inputs are not released. The paper reports the
  Qwen-native port reproduced in x87 stage 0.

## Paper claims, figures and tables: result files and scripts

| Paper element | Claim / content | Result file(s) | Producing script(s) | Status |
|---|---|---|---|---|
| §2.1, delexicalization | Filtering suppresses one auditor, not the lexical signal. The 15k pool is at chance for words (0.51) but not for characters (0.62); growing to 20k restores separability (0.72). | `data/results/extend_pool_A3_auc.json`, `extend_pool_A4_auc.json`; context: `auc_pairings.json`, `combined_pool_auc.json`, `confusable_A_auc.json` | `mining/s11_extend_pool_A3.py`, `s12_extend_pool_A4.py`, `s2_auc.py`, `s3_combined_pool.py` (`confusable_A_auc.json`: producer not included) | Property of the pools, not of a model. The in-loop trajectory (0.98 → near chance) and the 0.70 rebound come from the run log, with no result file. |
| §2.2 and the insertion-workflow figure | One inserted harm word raises refusal +0.31 / +0.19 / +0.33 (start/middle/end); the detection gate does not moderate it. | `data/results/x83_keyword_insertion.json` | `x83_*.py` | Primary analysis frozen before outcomes |
| Insertion-distribution figure | Per-prompt refusal shifts by position | drawn from `data/runs/x83/x83_refusal_{O,S,M,E}.jsonl`, `x83_detect.json` | (plotting script not included) | |
| §3, path B: crossed decomposition, projection figure | A held-out shared boundary direction (+6.28 / +7.20) whose projection tracks the refusal change (ρ 0.65 / 0.63) | `data/results/x85_crossed_keyword_latent.json`, `x85_crossed_keyword_directions.npz`, `x85_crossed_keyword_levels.json` | `x85_*.py` | Observational. The projection figure is recomputed from the x85 shards (not released; regenerable). |
| Heatmap figure ("where the reaction lives") | Word identity dominates at the word token; the refusal association appears at the boundary tokens from mid-depth on | `data/results/x85_crossed_keyword_latent_allstates.json` (+ `_state0_instruction.json`) | `x85_analyze.py` with `--states/--positions` (see x85) | Secondary, descriptive |
| §3, causal intervention | Subtracting lowers refusal 0.080, adding raises it 0.086; dose-ordered; beats KL-matched nulls; final `"\n\n"` token carries it; `comply2` mirrors; benign cost +0.058 | `data/results/x86_boundary_intervention.json` | `x86_*.py` | Pre-registered gates G0-G3 pass |
| §3 and appendix, path A | Refusal-weighted probe compiled into a rank-one `down_proj` edit (Qwen-native port) | `data/runs/x87/x75_qwen_*`, `x87_x75_qwen_config.json`, `x87_x75_qwen_dose_kl.json` | `x71_prepare_dataset.py`, `x75_prepare_weighted_probe.py`, `x75_train_weighted_probe.py`, `x75_train_edit.py`, `lib/edit_rank1.py` | Method; the two paths converge in position and behavior (no cosine between them is measured) |
| Crossed-design figure (appendix) | Schematic of the 96 × 24 design and folds | design in `data/runs/x85/x85_cohort.json` | `x85_build_cohort.py` | |
| §4 and the external-evaluation table | Hard-1K refusal −0.032 (write −0.5) / −0.078 (rank-one −3), superiority passes; guards mixed (four of six bounds outside 0.02); sign control +3 mirrors | `data/results/x87_external_safety_utility.json`; the write +0.5 column comes from `x89_positive_dose_extension.json` | `x87_*.py`, `x89_analyze.py` | Pre-registered (x87 conditions); the +0.5 column is descriptive |
| Dose-curve figure | Dose response of both families on the four endpoints, both signs | `x87_external_safety_utility.json`, `x89_positive_dose_extension.json`, `x87_gsm8k_posthoc.json` | `x87_analyze.py`, `x89_analyze.py`, `x87_gsm8k_analyze.py` | Deep doses, positive doses and GSM8K are post hoc |
| §5 and the transfer figure | The write at −1.5 lowers Hard-1K on 8/8 models with small costs; the rank-one edit at −6 (and −3) overshoots off its home model | `data/results/x88_cross_model_deep_dose.json` | `x88_*.py` | Exploratory, `claimable: false` |
| Supporting (not in the headline) | Removing a discovery-fitted word: inconclusive; ignore-instructions raise refusal | `x81_lexical_trigger.json`, `x81_posthoc_fragment_sensitivity.json`, `x81_de_self_recovery.json` | `x81*.py` | Supporting context |
| Not evidence | Paired refusal/compliance probes; primary gate failed | `x80_label_audit.json`, `x80_probe_geometry.json` | `x80_*.py` | Do not cite |

## Running the tests

```bash
python code/tests/run_tests.py
```

If `pytest` is installed, the runner delegates to it (`pytest -q code/tests`);
otherwise it loops over the plain assertion tests. The suite has 51 tests
across `test_edit_rank1`, `test_x75_weighting`, `test_x81_lexical_trigger`,
`test_x81_de`, `test_x83_keyword_insertion`, `test_x85_crossed_keyword`,
`test_x86_boundary_intervention` and `test_x87_common`. It runs on CPU in a
few seconds, with no GPU, model download or API key, and covers the rank-one
mechanics, probe weighting, insertion rules, cohort and fold construction,
hook arithmetic, bootstrap helpers and adapter validation. Some tests
parametrize over cases, so pytest reports more test cases than test
functions (58 passed in our check).

## Notes on the Slurm `.sbatch` files

`code/x80.sbatch`, `x81.sbatch`, `x83.sbatch`, `x85.sbatch` and `x86.sbatch`
are the wrappers used for the GPU stages. x87-x89 had no wrapper in this
release; run their commands directly or wrap them the same way. Each file
dispatches on environment variables:

| File | Variables | Stages |
|---|---|---|
| `x80.sbatch` | `SEED`, optional `EXT=1` | paired probes for one seed (`EXT` = extended L2 grid) |
| `x81.sbatch` | `STAGE` | `fills`, `semantics`, `gen_A`/`gen_B`/`gen_C`, `build_de`, `gen_P`/`gen_D`/`gen_E` |
| `x83.sbatch` | `STAGE` | `detect`, `gen_O`/`gen_S`/`gen_M`/`gen_E` |
| `x85.sbatch` | `STAGE`, `SHARD` (defaults to `SLURM_ARRAY_TASK_ID`), `NSHARDS` (default 24) | `extract`, `gen_O`/`gen_H`/`gen_N` |
| `x86.sbatch` | `STAGE`, `CONDITION` | `manifest`, `calibrate`, `generate`, `smoke` (4-row test) |

For example: `sbatch --time=04:00:00 --export=ALL,STAGE=extract --array=0-23 code/x85.sbatch`.

**Before submitting, adapt each file to your cluster:**

- Replace the `#SBATCH -A <your-slurm-account>` placeholder, add the
  partition (`-p`) and GPU type you use, and adjust the `-o` log path
  (default `logs/` relative to the submit directory; create it first).
- Export `B` (the repository root; required), `PY` (the Python interpreter,
  default `python`) and optionally `HF_HOME`; `x80.sbatch` also needs
  `SWSE_FC`. For example:
  `sbatch --export=ALL,B=$PWD,PY=$PWD/.venv/bin/python,STAGE=detect code/x83.sbatch`.
- **Pass `--time` explicitly.** Some clusters' short partitions default to
  15 minutes, which kills generation jobs.
- **Keep all `#SBATCH` lines in one contiguous block at the top of the
  file.** Slurm silently ignores directives that come after the first shell
  command. One submission wave of the original runs was lost to jobs
  falling back to default memory this way.
- The files set `HF_HUB_OFFLINE=1`, so models and datasets must already be
  in the cache (see [Offline compute nodes](#environment-setup)).
- Generators resume by id from their `--out` file. Use fresh output paths
  for an independent rerun.

## Citation

```bibtex
@inproceedings{lexicalfear2026,
  title     = {Language Models ``Fear'' Harmful Words: Causally Reducing a Lexically Triggered Driver of Over-Refusal},
  author    = {Zhang, Boyuan and Yigit, Ata Dundar and Zandsalimy, Mohammad and Sushmita, Shanu},
  booktitle = {NeurIPS 2026 Workshop on Foundations of Language Model Security},
  year      = {2026},
  note      = {Code and data: https://github.com/zboyr/lexical-fear, https://huggingface.co/datasets/vhboyr/lexical-fear}
}
```

## License

- **Code** (`code/`): MIT License; see [`LICENSE`](LICENSE).
- **Data** (`data/`, here and on Hugging Face): [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/),
  for non-commercial research use, subject to the content warning above.

The data includes prompts derived from third-party datasets (WildJailbreak,
OR-Bench, FalseReject, PHTest, CoCoNot, XSTest, JailbreakBench, ToxicChat,
Salad-Data, ALERT, StrongREJECT) and outputs of third-party models. Those
remain under their original licenses and terms of use.
