#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -o logs/%x_%j.out
# x83 GPU stages (shares the x81 tree). env STAGE: detect | gen_O | gen_S | gen_M | gen_E
set -eu
# B: release-repository root; PY: interpreter of the environment set up per README.md.
B=${B:?set B to the repository root}
PY=${PY:-python}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd $B/code
case "$STAGE" in
  detect) exec $PY x83_detect.py ;;
  gen_O|gen_S|gen_M|gen_E) exec $PY x83_generate.py --condition "${STAGE#gen_}" ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
