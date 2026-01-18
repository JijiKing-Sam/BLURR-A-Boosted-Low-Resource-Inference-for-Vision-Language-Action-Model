# open-pi-zero (vendored subset)

This directory vendors a minimal subset of the `open-pi-zero` project for the
BLURR demo/benchmark code release.

- Upstream: https://github.com/allenzren/open-pi-zero
- License: MIT (see `LICENSE`)

Included:
- Pi0 inference implementation (`src/model/**`)
- SimplerEnv evaluation agent + adapters (`src/agent/eval.py`, `src/agent/env_adapter/**`)
- Minimal configs for eval (`config/eval/**`, `config/*_statistics.json`)
- A sample image used by benchmarking scripts (`media/maniskill_pp.png`)

Not included:
- Training / dataset pipelines
- bitsandbytes-backed LoRA / 4-bit layers (this repo ships a minimal `src/model/lora.py`
  stub sufficient for the default non-LoRA configs)

