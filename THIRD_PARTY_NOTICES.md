# Third-Party Notices

This repository includes and/or integrates with third-party software. License
texts are included where required.

## Vendored code

### open-pi-zero (subset)

- Source: https://github.com/allenzren/open-pi-zero
- License: MIT
- Location in this repo: `third_party/open_pi_zero/`
- License file: `third_party/open_pi_zero/LICENSE`

Notes: We vendor a minimal subset needed for inference + SimplerEnv evaluation.
Training/data pipelines and bitsandbytes-backed LoRA/4-bit layers are not
included in this minimal release.

## External (not vendored)

### SimplerEnv

- Source: https://github.com/simpler-env/SimplerEnv
- Used for: evaluation environment (Simp lerEnv Bridge tasks) in BLURR demos.

