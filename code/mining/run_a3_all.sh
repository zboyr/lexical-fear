#!/bin/bash
# Pool A3 extension: build A3 (5k+5k, TF-IDF check) -> SWSE-label its harmful
# side -> merge into the 15k+15k set (data/pools/cp_15k_clustered.json and
# cp_15k_benign.json).
# t1/t2 are resume-safe; rerun this script to continue after a crash.
# PY: interpreter of the environment set up per README.md (default: python).
#
# Release note: the original script continued with Llama-3.2-3B feature
# extraction (t3), a 90/10 split (t4), train_probes.py, t5_eval.py and a
# concat-layer sweep, producing confusable_A15k_auc.json and
# concat_sweep_15k.json. Those outputs are not part of this release or the
# paper, so those steps are omitted here. The paper's harmful pool is the
# re-judged cp_15k_clustered_v2.json: run code/mining/rejudge_validated.py on
# the merged file afterwards (see README, "Upstream labelling chain").
set -e
cd "$(dirname "$0")/../.."
PY=${PY:-python}
M=code/mining
P=data/pools

$PY $M/s11_extend_pool_A3.py
$PY $M/t14_prepare_a3.py
$PY $M/t1_generate_responses.py --in $P/cp_a3_prompts.json --out $P/cp_a3_responses.json
$PY $M/t2_cluster.py --in $P/cp_a3_responses.json --out $P/cp_a3_clustered.json
$PY $M/t15_prepare_full15k.py
echo "ALL DONE"
