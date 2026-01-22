#!/usr/bin/env bash
# OpenVLA-OFT Bridge success on V100 (FP16 baseline).
#
# This is primarily for quick correctness checks (V100 lacks BF16 and FlashAttention2).
#
# Usage:
#   EPISODES=10 SEED=42 sbatch scripts/paper/run_bridge_success_v100_oft_only.sbatch

#SBATCH --job-name=blurr_bridge_success_oft_v100
#SBATCH --partition=GPU-shared
#SBATCH --account=cis250100p
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --time=02:30:00
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

EPISODES="${EPISODES:-10}"
SEED="${SEED:-42}"
OFT_MODEL_ID="${OFT_MODEL_ID:-szang18/openvla-oft-bridge-150k}"
OFT_TAG="${OFT_TAG:-${OFT_MODEL_ID//\//_}}"
OFT_TAG="${OFT_TAG//:/_}"

TASKS=(
  widowx_carrot_on_plate
  widowx_spoon_on_towel
  widowx_stack_cube
  widowx_put_eggplant_in_basket
)

OUT_ROOT="${OUT_ROOT:-${WORKSPACE_ROOT}/blurr_paper_results/bridge_success_v100_oft_${SEED}_${EPISODES}}"
mkdir -p "${OUT_ROOT}"

echo "Repo: ${REPO_ROOT}"
echo "Workspace: ${WORKSPACE_ROOT}"
echo "Episodes/task: ${EPISODES}"
echo "Seed: ${SEED}"
echo "Out: ${OUT_ROOT}"
echo "OFT model: ${OFT_MODEL_ID}"
echo "OFT tag: ${OFT_TAG}"

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis250100p/xma8/conda_envs/pi0

export HF_HOME="${HF_HOME:-${WORKSPACE_ROOT}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MS2_REAL2SIM_ASSET_DIR="${MS2_REAL2SIM_ASSET_DIR:-${WORKSPACE_ROOT}/SimplerEnv/ManiSkill2_real2sim/data}"

cd "${REPO_ROOT}"

echo "==== OpenVLA-OFT (${OFT_MODEL_ID}) baseline (fp16 eager) ===="
python -u scripts/eval_hf_vla_simpler.py \
  --model-id "${OFT_MODEL_ID}" \
  --preset baseline \
  --use-fp16 --no-torch-compile \
  --use-lora off \
  --task "${TASKS[@]}" \
  --seed "${SEED}" \
  --gpu-id 0 \
  --n-eval-episode "${EPISODES}" \
  --log-dir "${OUT_ROOT}/oft_${OFT_TAG}_fp16_eager"

echo "All Bridge OFT success runs finished. Summaries in: ${OUT_ROOT}"
