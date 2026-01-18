#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CHECKPOINT="${1:-}"
if [[ -z "${CHECKPOINT}" ]]; then
  echo "Usage: $0 /path/to/pi0_checkpoint.pt"
  exit 2
fi

# Optional (recommended) caches
export HF_HOME="${HF_HOME:-$ROOT/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

# Optional (SimplerEnv / ManiSkill assets)
# You likely need to set this to your SimplerEnv clone:
#   export MS2_REAL2SIM_ASSET_DIR=/path/to/SimplerEnv/ManiSkill2_real2sim/data
export MS2_REAL2SIM_ASSET_DIR="${MS2_REAL2SIM_ASSET_DIR:-}"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

EPISODES="${EPISODES:-100}"
VIDEOS="${VIDEOS:-0}"

TASKS=(
  widowx_spoon_on_towel
  widowx_carrot_on_plate
  widowx_stack_cube
  widowx_put_eggplant_in_basket
)

for TASK in "${TASKS[@]}"; do
  echo "===================== TASK: ${TASK} ====================="

  echo "[1] baseline (no prefix cache, fp32, steps=10)"
  python -u "$ROOT/scripts/eval_pi0_simpler.py" \
    --preset baseline \
    --config config/eval/bridge.yaml \
    --task "$TASK" \
    --checkpoint "$CHECKPOINT" \
    --n-eval-episode "$EPISODES" \
    --n-video "$VIDEOS"

  echo "[2] BLURR (prefix cache, bf16+compile, steps=1)"
  python -u "$ROOT/scripts/eval_pi0_simpler.py" \
    --preset blurr \
    --config config/eval/bridge.yaml \
    --task "$TASK" \
    --checkpoint "$CHECKPOINT" \
    --n-eval-episode "$EPISODES" \
    --n-video "$VIDEOS"
done

echo "Done. Aggregate with:"
echo "  python scripts/collect_bridge_eval_results.py"

