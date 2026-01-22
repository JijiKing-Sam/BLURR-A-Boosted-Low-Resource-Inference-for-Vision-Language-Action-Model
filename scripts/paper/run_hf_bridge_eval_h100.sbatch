#!/usr/bin/env bash
#SBATCH --job-name=blurr_hf_bridge_eval_h100
#SBATCH --partition=GPU-shared
#SBATCH --account=cis250100p
#SBATCH --qos=gpuinteract
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
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

EPISODES="${EPISODES:-100}"
SEED="${SEED:-42}"
OFT_MODEL_ID="${OFT_MODEL_ID:-szang18/openvla-oft-bridge-150k}"

TASKS=(
  widowx_spoon_on_towel
  widowx_carrot_on_plate
  widowx_stack_cube
  widowx_put_eggplant_in_basket
)

echo "Repo: ${REPO_ROOT}"
echo "Workspace: ${WORKSPACE_ROOT}"
echo "Episodes: ${EPISODES}"
echo "Seed: ${SEED}"
echo "OpenVLA-OFT model: ${OFT_MODEL_ID}"

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis250100p/xma8/conda_envs/pi0

export HF_HOME="${HF_HOME:-${WORKSPACE_ROOT}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export MS2_REAL2SIM_ASSET_DIR="${MS2_REAL2SIM_ASSET_DIR:-${WORKSPACE_ROOT}/SimplerEnv/ManiSkill2_real2sim/data}"

cd "${REPO_ROOT}"

MODELS_BASELINE=(
  openvla/openvla-7b
  "${OFT_MODEL_ID}"
)

echo "==== Baseline HF models (FP32 eager) ===="
for MODEL in "${MODELS_BASELINE[@]}"; do
  EXTRA_LORA=()
  if [[ "${MODEL}" == "${OFT_MODEL_ID}" ]]; then
    EXTRA_LORA=( --use-lora off )
  fi
  for TASK in "${TASKS[@]}"; do
    python -u scripts/eval_hf_vla_simpler.py \
      --model-id "${MODEL}" \
      --preset baseline \
      --task "${TASK}" \
      --seed "${SEED}" \
      --gpu-id 0 \
      --n-eval-episode "${EPISODES}" \
      "${EXTRA_LORA[@]}"
  done
done

echo "==== OpenVLA-OFT + BLURR (BF16 + compile) ===="
for TASK in "${TASKS[@]}"; do
  python -u scripts/eval_hf_vla_simpler.py \
    --model-id "${OFT_MODEL_ID}" \
    --preset blurr \
    --use-lora off \
    --task "${TASK}" \
    --seed "${SEED}" \
    --gpu-id 0 \
    --n-eval-episode "${EPISODES}"
done

echo "All HF Bridge evaluations finished."
