#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -o logs/%x_%j.out
# x81 GPU stages. env STAGE: fills | semantics | gen_A | gen_B | gen_C | build_de | gen_P | gen_D | gen_E
set -eu
# B: release-repository root; PY: interpreter of the environment set up per README.md.
B=${B:?set B to the repository root}
PY=${PY:-python}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
case "$STAGE" in
  fills)     exec $PY $B/code/x81_score_fills.py ;;
  semantics) exec $PY $B/code/x81_judge_semantics.py ;;
  build_de) exec $PY $B/code/x81_build_arms_de.py ;;
  gen_A|gen_B|gen_C) exec $PY $B/code/x81_generate.py --condition "${STAGE#gen_}" ;;
  gen_P|gen_D|gen_E) exec $PY $B/code/x81_generate.py --condition "${STAGE#gen_}" --arms $B/data/runs/x81/x81_arms_de.json ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
