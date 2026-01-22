#!/usr/bin/env bash
#SBATCH --job-name=blurr_hf_microbench_h100
#SBATCH --partition=GPU-shared
#SBATCH --account=cis250100p
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=5
#SBATCH --time=02:00:00
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

# Override at submit time if needed.
HF_MODELS="${HF_MODELS:-openvla/openvla-7b Kaipengm2/openvla-oft-64-130000 Stanford-ILIAD/minivla-vq-bridge-prismatic}"
HF_PROFILES="${HF_PROFILES:-bf16_eager bf16_compile}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-50}"

echo "Repo: ${REPO_ROOT}"
echo "Workspace: ${WORKSPACE_ROOT}"
echo "Output: ${OUT_DIR}"
echo "HF_MODELS: ${HF_MODELS}"
echo "HF_PROFILES: ${HF_PROFILES}"
echo "WARMUP: ${WARMUP} | ITERS: ${ITERS}"

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis250100p/xma8/conda_envs/pi0

export HF_HOME="${HF_HOME:-${WORKSPACE_ROOT}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "${REPO_ROOT}"

python -u scripts/paper/hf_microbench.py \
  --out-csv "${OUT_DIR}/hf_microbench_h100_extra.csv" \
  --model-id ${HF_MODELS} \
  --profiles ${HF_PROFILES} \
  --warmup "${WARMUP}" --iters "${ITERS}"

echo "Done. Wrote: ${OUT_DIR}/hf_microbench_h100_extra.csv"
