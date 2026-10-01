#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -o logs/%x_%j.out
# x85 GPU stages. env STAGE=extract|gen_O|gen_H|gen_N; extract takes SHARD (default: the
# zero-based array index SLURM_ARRAY_TASK_ID) and NSHARDS (default 24). Pass --time at
# submit: devel partitions default to a 15-minute limit.
set -euo pipefail
# B: release-repository root; PY: interpreter of the environment set up per README.md.
B=${B:?set B to the repository root}
PY=${PY:-python}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd "$B/code"
case "$STAGE" in
  extract)
    SHARD=${SHARD:-${SLURM_ARRAY_TASK_ID:?set SHARD or submit as an array}}
    NSHARDS=${NSHARDS:-24}
    exec "$PY" x85_extract_hidden.py --shard "$SHARD" --num-shards "$NSHARDS"
    ;;
  gen_O|gen_H|gen_N)
    exec "$PY" x85_generate.py --condition "${STAGE#gen_}"
    ;;
  *)
    echo "unknown STAGE=$STAGE" >&2
    exit 2
    ;;
esac
