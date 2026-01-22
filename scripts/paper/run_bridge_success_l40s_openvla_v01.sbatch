#!/usr/bin/env bash
# OpenVLA v0.1 Bridge success on L40S (baseline vs BLURR-style inference flags).
#
# Usage:
#   EPISODES=10 SEED=42 sbatch scripts/paper/run_bridge_success_l40s_openvla_v01.sbatch
#
# Optional extra eval args (passed after the script path):
#   sbatch scripts/paper/run_bridge_success_l40s_openvla_v01.sbatch --center-crop off
#
# Notes:
# - OpenVLA v0.1 uses a Vicuna base model and expects a system prompt + "USER: ... ASSISTANT:" format.

#SBATCH --job-name=blurr_bridge_success_openvla_v01_l40s
#SBATCH --partition=GPU-shared
#SBATCH --account=cis250100p
#SBATCH --qos=gpuinteract
#SBATCH --gres=gpu:l40s-48:1
#SBATCH --cpus-per-task=5
#SBATCH --time=03:00:00
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

EPISODES="${EPISODES:-10}"
SEED="${SEED:-42}"
MODEL_ID="${MODEL_ID:-openvla/openvla-v01-7b}"
MODEL_TAG="${MODEL_TAG:-${MODEL_ID//\//_}}"
MODEL_TAG="${MODEL_TAG//:/_}"

TASKS=(
  widowx_carrot_on_plate
  widowx_spoon_on_towel
  widowx_stack_cube
  widowx_put_eggplant_in_basket
)

EXTRA_ARGS=( "$@" )

OUT_ROOT="${OUT_ROOT:-${WORKSPACE_ROOT}/blurr_paper_results/bridge_success_l40s_${MODEL_TAG}_${SEED}_${EPISODES}}"
mkdir -p "${OUT_ROOT}"

SYSTEM_PROMPT="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
PROMPT_TEMPLATE="${SYSTEM_PROMPT} USER: What action should the robot take to {instruction}? ASSISTANT:"

echo "Repo: ${REPO_ROOT}"
echo "Workspace: ${WORKSPACE_ROOT}"
echo "Episodes/task: ${EPISODES}"
echo "Seed: ${SEED}"
echo "Out: ${OUT_ROOT}"
echo "Model: ${MODEL_ID}"
echo "Extra args: ${EXTRA_ARGS[*]}"

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis250100p/xma8/conda_envs/pi0

export HF_HOME="${HF_HOME:-${WORKSPACE_ROOT}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MS2_REAL2SIM_ASSET_DIR="${MS2_REAL2SIM_ASSET_DIR:-${WORKSPACE_ROOT}/SimplerEnv/ManiSkill2_real2sim/data}"

cd "${REPO_ROOT}"

echo "==== OpenVLA v0.1 (${MODEL_ID}) baseline (bf16 eager) ===="
python -u scripts/eval_hf_vla_simpler.py \
  --model-id "${MODEL_ID}" \
  --preset baseline \
  --use-bf16 --no-torch-compile \
  --prompt-template "${PROMPT_TEMPLATE}" \
  --task "${TASKS[@]}" \
  --seed "${SEED}" \
  --gpu-id 0 \
  --n-eval-episode "${EPISODES}" \
  --log-dir "${OUT_ROOT}/openvla_v01_${MODEL_TAG}_bf16_eager" \
  "${EXTRA_ARGS[@]}"

echo "==== OpenVLA v0.1 (${MODEL_ID}) + BLURR (bf16 + flash_attention_2 + torch.compile) ===="
python -u scripts/eval_hf_vla_simpler.py \
  --model-id "${MODEL_ID}" \
  --preset blurr \
  --attn-implementation flash_attention_2 \
  --prompt-template "${PROMPT_TEMPLATE}" \
  --task "${TASKS[@]}" \
  --seed "${SEED}" \
  --gpu-id 0 \
  --n-eval-episode "${EPISODES}" \
  --log-dir "${OUT_ROOT}/openvla_v01_${MODEL_TAG}_blurr" \
  "${EXTRA_ARGS[@]}"

echo "All Bridge OpenVLA v0.1 runs finished. Summaries in: ${OUT_ROOT}"

