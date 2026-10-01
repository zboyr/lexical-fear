#!/usr/bin/env bash
# Pilot chain behind data/results/confusable_A_auc.json: stages 2-5 after t1
# (response generation for the 1,000-harmful + 1,000-benign pilot from t0).
# All stages resume-safe. Run t0_prepare.py and t1_generate_responses.py first.
# PY: interpreter of the environment set up per README.md (default: python).
set -e
cd "$(dirname "$0")/../.."
PY=${PY:-python}
M=code/mining

echo "===== waiting for t1 (1000 harmful responses) ====="
until $PY -c "import json,sys; d=json.load(open('data/pools/cp_responses.json')); n=sum('llm_responses' in x for x in d); print('have',n,flush=True); sys.exit(0 if n>=1000 else 1)"; do
  sleep 30
done
echo "t1 complete"

echo "===== t2: cluster harmful -> T ====="
$PY $M/t2_cluster.py

echo "===== t3: extract TBG features (2000) ====="
$PY $M/t3_extract_features.py --cache_dir feature_caches/confusable_A_full --device cuda

echo "===== t4: split 70/30 ====="
$PY $M/t4_split.py

echo "===== train: official MLP + linear probes ====="
$PY $M/train_probes.py --cache_dir feature_caches/confusable_A_train --output_dir cpA \
    --concat_layers 16 20 25 27 28 --alpha 0.7 --device cuda

RUN=$(ls -d runs/cpA* 2>/dev/null | sort -V | tail -1)
echo "trained run: $RUN"

echo "===== t5: evaluate all predictors ====="
$PY $M/t5_eval.py --run "$RUN" \
    --full_cache feature_caches/confusable_A_full --test_cache feature_caches/confusable_A_test

echo "===== DONE ====="
