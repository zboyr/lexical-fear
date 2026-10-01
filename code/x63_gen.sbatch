#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH --job-name=x63-gen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x_%j.out
# x63 10-sample generation for one row shard of one model.
# env MODEL_ID (HF id or local snapshot), TAG (model tag, e.g. qwen), START, END (row range);
# optional BATCH (default 16) and LIMIT (smoke test: first N rows of the shard).
set -euo pipefail

: "${MODEL_ID:?submit with --export=ALL,MODEL_ID=...,TAG=...,START=...,END=...}"
: "${TAG:?}"
: "${START:?}"
: "${END:?}"

# B: release-repository root; PY: interpreter of the environment set up per README.md.
B=${B:?set B to the repository root}
PY=${PY:-python}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1

mkdir -p "${B}/data/generations"
exec "$PY" "${B}/code/x63_gen_hpc.py" \
  --model "${MODEL_ID}" \
  --in "${B}/data/prompts/x63_prompts13527.json" \
  --out "${B}/data/generations/x63_${TAG}_${START}_${END}.json" \
  --start "${START}" --end "${END}" \
  --batch "${BATCH:-16}" ${LIMIT:+--limit "$LIMIT"}
