#!/usr/bin/env bash
# MiniVLA (prismatic .pt checkpoint) Bridge success on L40S.
#
# Usage:
#   EPISODES=50 SEED=42 sbatch scripts/paper/run_bridge_success_l40s_minivla_prismatic.sbatch
#
# Notes:
# - This evaluates a Prismatic/OpenVLA-format `.pt` checkpoint (MiniVLA VQ) using `scripts/eval_prismatic_vla_simpler.py`.
# - Requires an `openvla-mini` checkout at `${WORKSPACE_ROOT}/openvla-mini` (default in this workspace).

#SBATCH --job-name=blurr_bridge_success_minivla_l40s
#SBATCH --partition=GPU-shared
#SBATCH --account=cis250100p
#SBATCH --qos=gpuinteract
#SBATCH --gres=gpu:l40s-48:1
#SBATCH --cpus-per-task=5
#SBATCH --time=08:00:00
#SBATCH --export=ALL
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -d "${SUBMIT_DIR}/blurr" ]]; then
  REPO_ROOT="${SUBMIT_DIR}"
elif [[ -d "${SUBMIT_DIR}/BLURR-A-Boosted-Low-Resource-Inference-for-Vision-Language-Action-Model/blurr" ]]; then
  REPO_ROOT="${SUBMIT_DIR}/BLURR-A-Boosted-Low-Resource-Inference-for-Vision-Language-Action-Model"
else
  echo "ERROR: Could not locate BLURR repo root from SLURM_SUBMIT_DIR='${SUBMIT_DIR}'" >&2
  exit 2
fi
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

EPISODES="${EPISODES:-50}"
SEED="${SEED:-42}"

TASKS=(
  widowx_carrot_on_plate
  widowx_spoon_on_towel
  widowx_stack_cube
  widowx_put_eggplant_in_basket
)

OUT_ROOT="${OUT_ROOT:-${WORKSPACE_ROOT}/blurr_paper_results/bridge_success_l40s_minivla_${SEED}_${EPISODES}}"
mkdir -p "${OUT_ROOT}"

MINIVLA_CKPT="${MINIVLA_CKPT:-${WORKSPACE_ROOT}/hf_cache/hub/models--Stanford-ILIAD--minivla-vq-bridge-prismatic/snapshots/931a637fbfc783220df9c47eb613bf6c9c6e1c4a/checkpoints/step-362500-epoch-21-loss=0.2259.pt}"
OPENVLA_MINI_ROOT="${OPENVLA_MINI_ROOT:-${WORKSPACE_ROOT}/openvla-mini}"

echo "Repo: ${REPO_ROOT}"
echo "Workspace: ${WORKSPACE_ROOT}"
echo "Episodes/task: ${EPISODES}"
echo "Seed: ${SEED}"
echo "Out: ${OUT_ROOT}"
echo "MiniVLA ckpt: ${MINIVLA_CKPT}"
echo "openvla-mini root: ${OPENVLA_MINI_ROOT}"

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis250100p/xma8/conda_envs/pi0

export HF_HOME="${HF_HOME:-${WORKSPACE_ROOT}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MS2_REAL2SIM_ASSET_DIR="${MS2_REAL2SIM_ASSET_DIR:-${WORKSPACE_ROOT}/SimplerEnv/ManiSkill2_real2sim/data}"

cd "${REPO_ROOT}"

echo "==== MiniVLA (prismatic checkpoint) ===="
python -u scripts/eval_prismatic_vla_simpler.py \
  --checkpoint "${MINIVLA_CKPT}" \
  --openvla-mini-root "${OPENVLA_MINI_ROOT}" \
  --task "${TASKS[@]}" \
  --seed "${SEED}" \
  --gpu-id 0 \
  --n-eval-episode "${EPISODES}" \
  --log-dir "${OUT_ROOT}/minivla_prismatic"

echo "All Bridge MiniVLA runs finished. Summary in: ${OUT_ROOT}/minivla_prismatic/summary.json"

