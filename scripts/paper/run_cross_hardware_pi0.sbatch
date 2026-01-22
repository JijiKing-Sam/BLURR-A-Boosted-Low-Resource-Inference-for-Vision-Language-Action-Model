#!/usr/bin/env bash
#SBATCH --job-name=blurr_hw_pi0
#SBATCH --partition=GPU-shared
#SBATCH --account=cis250100p
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
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

OUT_DIR="${OUT_DIR:-${WORKSPACE_ROOT}/blurr_paper_results/cross_hardware}"
mkdir -p "${OUT_DIR}"

CHECKPOINT="${CHECKPOINT:-/ocean/projects/cis250100p/xma8/pi0_setup/open-pi-zero/results/pretrained/bridge_beta_step19296_2024-12-26_22-30_42.pt}"

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis250100p/xma8/conda_envs/pi0

export HF_HOME="${HF_HOME:-${WORKSPACE_ROOT}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "${REPO_ROOT}"

python -u scripts/paper/pi0_microbench.py compare-presets \
  --config config/eval/bridge.yaml \
  --checkpoint "${CHECKPOINT}" \
  --out-json "${OUT_DIR}/pi0_${SLURM_JOB_ID}.json" \
  --presets baseline blurr \
  --warmup 5 --iters 50 \
  --skip-flops

echo "Wrote cross-hardware result to: ${OUT_DIR}/pi0_${SLURM_JOB_ID}.json"
