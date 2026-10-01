#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=3:00:00
#SBATCH -o logs/%x_%j.out
# x80 Stages 1-2: paired probes. env: SEED
set -eu
# B: release-repository root; PY: interpreter of the environment set up per README.md.
B=${B:?set B to the repository root}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
"${PY:-python}" $B/code/x80_fit_paired_probes.py \
  --cache "${SWSE_FC:?set SWSE_FC}/qwen_A15k_v4_full/hidden_states.pt" \
  --labels $B/data/runs/x80/x80_labels.npz --indices-dir $B/data/runs/x66 \
  --seed $SEED --out-dir $B/data/runs/x80/probes${EXT:+_l2ext} --n-perm 5 --n-boot 50 ${EXT:+--l2-ext}
