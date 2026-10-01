#!/bin/bash
# Label the remaining 4000 harmful prompts of pool A with the SWSE pipeline:
# t9 (prepare) -> t1 (sample 10 responses, Llama-3.2-3B) -> t2 (local Qwen judge -> T).
# Both t1 and t2 are resume-safe; rerun this script to continue after a crash.
# PY: interpreter of the environment set up per README.md (default: python).
set -e
cd "$(dirname "$0")/../.."
PY=${PY:-python}
M=code/mining
P=data/pools

$PY $M/t9_prepare_rest.py
$PY $M/t1_generate_responses.py --in $P/cp_rest_prompts.json --out $P/cp_rest_responses.json
$PY $M/t2_cluster.py --in $P/cp_rest_responses.json --out $P/cp_rest_clustered.json
echo "ALL DONE"
