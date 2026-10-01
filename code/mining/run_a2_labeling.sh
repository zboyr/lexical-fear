#!/bin/bash
# SWSE-label pool A2's 5000 harmful prompts:
# t12 (prepare) -> t1 (sample 10 responses, Llama-3.2-3B) -> t2 (Qwen judge -> T).
# Resume-safe; rerun to continue after a crash.
# PY: interpreter of the environment set up per README.md (default: python).
set -e
cd "$(dirname "$0")/../.."
PY=${PY:-python}
M=code/mining
P=data/pools

$PY $M/t12_prepare_a2.py
$PY $M/t1_generate_responses.py --in $P/cp_a2_prompts.json --out $P/cp_a2_responses.json
$PY $M/t2_cluster.py --in $P/cp_a2_responses.json --out $P/cp_a2_clustered.json
echo "ALL DONE"
