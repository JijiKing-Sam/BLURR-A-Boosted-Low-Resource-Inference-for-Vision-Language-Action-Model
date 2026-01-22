#!/usr/bin/env bash
#SBATCH --job-name=blurr_bridge_success_l40s
#SBATCH --partition=GPU-shared
#SBATCH --account=cis250100p
#SBATCH --qos=gpuinteract
#SBATCH --gres=gpu:l40s-48:1
#SBATCH --cpus-per-task=5
#SBATCH --time=06:00:00
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

EPISODES="${EPISODES:-20}"
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

OUT_ROOT="${OUT_ROOT:-${WORKSPACE_ROOT}/blurr_paper_results/bridge_success_l40s_${SEED}_${EPISODES}}"
mkdir -p "${OUT_ROOT}"

MINIVLA_CKPT="${MINIVLA_CKPT:-${WORKSPACE_ROOT}/hf_cache/hub/models--Stanford-ILIAD--minivla-vq-bridge-prismatic/snapshots/931a637fbfc783220df9c47eb613bf6c9c6e1c4a/checkpoints/step-362500-epoch-21-loss=0.2259.pt}"

echo "Repo: ${REPO_ROOT}"
echo "Workspace: ${WORKSPACE_ROOT}"
echo "Episodes/task: ${EPISODES}"
echo "Seed: ${SEED}"
echo "Out: ${OUT_ROOT}"
echo "MiniVLA ckpt: ${MINIVLA_CKPT}"
echo "OpenVLA-OFT model: ${OFT_MODEL_ID}"

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis250100p/xma8/conda_envs/pi0

export HF_HOME="${HF_HOME:-${WORKSPACE_ROOT}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MS2_REAL2SIM_ASSET_DIR="${MS2_REAL2SIM_ASSET_DIR:-${WORKSPACE_ROOT}/SimplerEnv/ManiSkill2_real2sim/data}"

cd "${REPO_ROOT}"

echo "==== OpenVLA (bf16 eager, SDPA default) ===="
python -u scripts/eval_hf_vla_simpler.py \
  --model-id openvla/openvla-7b \
  --preset baseline \
  --use-bf16 --no-torch-compile \
  --task "${TASKS[@]}" \
  --seed "${SEED}" \
  --gpu-id 0 \
  --n-eval-episode "${EPISODES}" \
  --log-dir "${OUT_ROOT}/openvla_bf16_eager"

echo "==== MiniVLA (prismatic checkpoint) ===="
python -u scripts/eval_prismatic_vla_simpler.py \
  --checkpoint "${MINIVLA_CKPT}" \
  --task "${TASKS[@]}" \
  --seed "${SEED}" \
  --gpu-id 0 \
  --n-eval-episode "${EPISODES}" \
  --log-dir "${OUT_ROOT}/minivla_prismatic"

echo "==== OpenVLA-OFT (bf16 eager, SDPA default) ===="
python -u scripts/eval_hf_vla_simpler.py \
  --model-id "${OFT_MODEL_ID}" \
  --preset baseline \
  --use-bf16 --no-torch-compile \
  --use-lora off \
  --task "${TASKS[@]}" \
  --seed "${SEED}" \
  --gpu-id 0 \
  --n-eval-episode "${EPISODES}" \
  --log-dir "${OUT_ROOT}/openvla_oft_${OFT_TAG}_bf16_eager"

echo "==== OpenVLA-OFT + BLURR (bf16 + flash_attention_2 + torch.compile) ===="
python -u scripts/eval_hf_vla_simpler.py \
  --model-id "${OFT_MODEL_ID}" \
  --preset blurr \
  --attn-implementation flash_attention_2 \
  --use-lora off \
  --task "${TASKS[@]}" \
  --seed "${SEED}" \
  --gpu-id 0 \
  --n-eval-episode "${EPISODES}" \
  --log-dir "${OUT_ROOT}/openvla_oft_${OFT_TAG}_blurr"

echo "All Bridge success runs finished. Summaries in: ${OUT_ROOT}"
