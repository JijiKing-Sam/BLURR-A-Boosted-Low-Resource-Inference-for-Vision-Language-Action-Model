#!/usr/bin/env bash
#SBATCH --job-name=blurr_openvla_oft_blurr_v100
#SBATCH --partition=GPU-small
#SBATCH --account=cis250100p
#SBATCH --qos=gpuinteract
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --time=01:00:00
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

OUT_DIR="${OUT_DIR:-${WORKSPACE_ROOT}/blurr_paper_results}"
mkdir -p "${OUT_DIR}"

echo "Repo: ${REPO_ROOT}"
echo "Workspace: ${WORKSPACE_ROOT}"
echo "Output: ${OUT_DIR}"

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis250100p/xma8/conda_envs/pi0

export HF_HOME="${HF_HOME:-${WORKSPACE_ROOT}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "${REPO_ROOT}"

echo "==== Baseline (SDPA default) ===="
python -u scripts/paper/hf_microbench.py \
  --out-csv "${OUT_DIR}/hf_microbench_v100_openvla_oft_baseline.csv" \
  --model-id openvla/openvla-7b Kaipengm2/openvla-oft-64-130000 \
  --profiles bf16_eager \
  --warmup 5 --iters 50

echo "==== OpenVLA-OFT + BLURR (FlashAttention-2 + compile) ===="
python -u scripts/paper/hf_microbench.py \
  --out-csv "${OUT_DIR}/hf_microbench_v100_openvla_oft_blurr.csv" \
  --model-id Kaipengm2/openvla-oft-64-130000 \
  --profiles bf16_compile \
  --attn-implementation flash_attention_2 \
  --warmup 5 --iters 50

echo "Done."
