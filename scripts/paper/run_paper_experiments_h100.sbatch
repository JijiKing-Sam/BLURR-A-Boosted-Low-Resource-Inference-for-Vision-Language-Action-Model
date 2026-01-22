#!/usr/bin/env bash
#SBATCH --job-name=blurr_paper_h100
#SBATCH --partition=GPU-shared
#SBATCH --account=cis250100p
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
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

CHECKPOINT="${CHECKPOINT:-/ocean/projects/cis250100p/xma8/pi0_setup/open-pi-zero/results/pretrained/bridge_beta_step19296_2024-12-26_22-30_42.pt}"

# Override these at submit time, e.g.:
#   TEXT_TOKENS="0 6 12 18" HF_PROFILES="bf16_eager bf16_compile" sbatch ...
TEXT_TOKENS="${TEXT_TOKENS:-0 6 12 18}"
HORIZON_STEPS="${HORIZON_STEPS:-1 2 4 10}"
HORIZON_EPISODES="${HORIZON_EPISODES:-10}"
HF_MODELS="${HF_MODELS:-openvla/openvla-7b Kaipengm2/openvla-oft-64-130000}"
HF_PROFILES="${HF_PROFILES:-bf16_eager bf16_compile}"

echo "Repo: ${REPO_ROOT}"
echo "Workspace: ${WORKSPACE_ROOT}"
echo "Output: ${OUT_DIR}"
echo "Checkpoint: ${CHECKPOINT}"

source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate /ocean/projects/cis250100p/xma8/conda_envs/pi0

export HF_HOME="${HF_HOME:-${WORKSPACE_ROOT}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export MS2_REAL2SIM_ASSET_DIR="${MS2_REAL2SIM_ASSET_DIR:-/ocean/projects/cis250100p/xma8/SimplerEnv/ManiSkill2_real2sim/data}"

cd "${REPO_ROOT}"

echo "==== [1/5] Prompt length sweep (KV cache on/off) ===="
python -u scripts/paper/pi0_microbench.py prompt-sweep \
  --config config/eval/bridge.yaml \
  --checkpoint "${CHECKPOINT}" \
  --out-csv "${OUT_DIR}/prompt_length_sweep_pi0.csv" \
  --text-tokens ${TEXT_TOKENS} \
  --num-inference-steps 10 \
  --warmup 5 --iters 50 \
  --skip-flops

echo "==== [2/5] First-call vs steady-state ===="
python -u scripts/paper/pi0_microbench.py first-vs-steady \
  --config config/eval/bridge.yaml \
  --checkpoint "${CHECKPOINT}" \
  --out-csv "${OUT_DIR}/first_vs_steady_pi0.csv" \
  --warmup 5 --iters 50

echo "==== [3/5] Latency vs num_inference_steps sweep ===="
python -u scripts/paper/pi0_microbench.py steps-sweep \
  --config config/eval/bridge.yaml \
  --checkpoint "${CHECKPOINT}" \
  --out-csv "${OUT_DIR}/steps_latency_pi0_bf16_compile.csv" \
  --steps 1 2 4 6 10 \
  --use-bf16 --use-torch-compile \
  --warmup 5 --iters 50 \
  --skip-flops

echo "==== [4/5] Success vs horizon sweep (SimplerEnv) ===="
python -u scripts/paper/horizon_sweep_simpler.py \
  --config config/eval/bridge.yaml \
  --checkpoint "${CHECKPOINT}" \
  --out-root "${OUT_DIR}/horizon_sweep_runs" \
  --out-csv "${OUT_DIR}/horizon_sweep_success.csv" \
  --steps ${HORIZON_STEPS} \
  --episodes ${HORIZON_EPISODES} \
  --disable-torch-compile

echo "==== [5/5] Cross-model HF microbench (OpenVLA + OFT) ===="
python -u scripts/paper/hf_microbench.py \
  --out-csv "${OUT_DIR}/hf_microbench.csv" \
  --model-id ${HF_MODELS} \
  --profiles ${HF_PROFILES} \
  --warmup 5 --iters 50 \
  --skip-flops

echo "All paper experiments finished. Results in: ${OUT_DIR}"
