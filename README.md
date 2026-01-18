# BLURR: A Boosted Low-Resource Inference for Vision-Language-Action Model

🚀 Interactive Demo Available! > Watch real-time comparisons and detailed visualization comparisons on our web demo:

👉 [https://jijiking-sam.github.io/BLURR.../demo/](https://jijiking-sam.github.io/BLURR-A-Boosted-Low-Resource-Inference-for-Vision-Language-Action-Model/demo/)

An open-source demo for VLA inference acceleration.

BLURR is a lightweight inference wrapper for Vision–Language–Action (VLA) controllers such as Pi-0 and OpenVLA.  
It **keeps the original checkpoints and APIs unchanged**, and accelerates control by:

- caching the **instruction prefix** with a KV cache,
- running the decoder in **BF16 with compiled kernels and FlashAttention**, and
- using a **single-step rollout schedule** instead of multi-step flows.

This repository provides the demo code and short video clips used in our WWW 2026 demo submission.

01.12 Paper has been accepted as a demo paper at the theWebConf2026 conference.

---

## 0. Reproducibility (Code)

This repository now includes runnable code to reproduce the BLURR inference stack and basic evaluations:

- Pi0 + SimplerEnv evaluation runner: `scripts/eval_pi0_simpler.py`
- Pi0 latency/VRAM/GFLOPS benchmark: `scripts/benchmark_pi0.py`
- HuggingFace VLA benchmark (e.g., OpenVLA): `scripts/benchmark_hf_vla.py`
- Bridge batch runner + result collector: `scripts/run_bridge_full_eval.sh`, `scripts/collect_bridge_eval_results.py`

We vendor a minimal subset of the open-pi-zero implementation under `third_party/open_pi_zero/` (MIT license).
The demo webpage is fully self-contained in `demo/index.html` (no build step).

### 0.1 Install

```bash
pip install -r requirements.txt
```

For SimplerEnv evaluation, also install SimplerEnv (and ManiSkill2_real2sim) from its repo and ensure the assets are available (see SimplerEnv docs).

### 0.2 Run Pi0 + SimplerEnv (single task)

```bash
python scripts/eval_pi0_simpler.py \
  --preset blurr \
  --config config/eval/bridge.yaml \
  --task widowx_carrot_on_plate \
  --checkpoint /path/to/bridge_beta_step19296_*.pt \
  --n-eval-episode 10
```

Logs go to `runs/eval_bridge/.../run.log`.

### 0.3 Run Bridge batch evaluation (4 tasks, baseline vs BLURR)

```bash
bash scripts/run_bridge_full_eval.sh /path/to/bridge_beta_step19296_*.pt
python scripts/collect_bridge_eval_results.py
```

### 0.4 Run benchmarks

Pi0 local checkpoint benchmark:

```bash
python scripts/benchmark_pi0.py \
  --config config/eval/bridge.yaml \
  --checkpoint /path/to/bridge_beta_step19296_*.pt \
  --use-bf16 --use-torch-compile \
  --warmup 5 --iters 50
```

HuggingFace VLA benchmark (example):

```bash
python scripts/benchmark_hf_vla.py \
  --model-id openvla/openvla-7b \
  --use-bf16 --use-torch-compile
```

---

## 1. Demo Clips

Below are side-by-side comparisons recorded from our web demo.
Left: Baseline Interleave-Pi-0 (~6Hz). Right: BLURR-Pi-0 (>50Hz).
Note the smoother control and faster reaction times in the BLURR column.

<table>
<tr>
<th width="50%">Baseline (Interleave-Pi-0)

<sub>High Latency (~162ms)</sub></th>

<th width="50%">BLURR (Ours)

<sub>Low Latency (~17ms)</sub></th>

</tr>

<!-- Task 1: Carrot -->

<tr>
<td colspan="2" align="center"><b>1. Carrot on Plate</b></td>
</tr>
<tr>
<td>
<video src="https://github.com/user-attachments/assets/82b17c6c-2520-4f85-b126-a5e23c8a7d3a" width="100%" autoplay loop muted playsinline></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/9013b868-d8c0-4a6b-a57b-eea829289a1e" width="100%" autoplay loop muted playsinline></video>
</td>
</tr>

<!-- Task 2: Spoon -->

<tr>
<td colspan="2" align="center"><b>2. Spoon on Cloth</b></td>
</tr>
<tr>
<td>
<video src="https://github.com/user-attachments/assets/85a1cc95-71a7-4aef-82e2-95004c905e5a" width="100%" autoplay loop muted playsinline></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/c8c5564a-79e6-42c2-a1dc-0de715956834" width="100%" autoplay loop muted playsinline></video>
</td>
</tr>

<!-- Task 3: Eggplant -->

<tr>
<td colspan="2" align="center"><b>3. Eggplant in Rack</b></td>
</tr>
<tr>
<td>
<video src="https://github.com/user-attachments/assets/ee19ee2e-7df7-4092-9f1d-c4009343cf33" width="100%" autoplay loop muted playsinline></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/2ecea4b1-ceb4-4f1d-b9b2-d8ffe10eebf1" width="100%" autoplay loop muted playsinline></video>
</td>
</tr>

<!-- Task 4: Blocks -->

<tr>
<td colspan="2" align="center"><b>4. Block Stacking</b></td>
</tr>
<tr>
<td>
<video src="https://github.com/user-attachments/assets/21cc8391-3758-4ba2-807e-477435832847" width="100%" autoplay loop muted playsinline></video>
</td>
<td>
<video src="https://github.com/user-attachments/assets/3d2275dc-5a89-40c1-a5d6-9aa6e9a75ba2" width="100%" autoplay loop muted playsinline></video>
</td>
</tr>
</table>

Note: If videos do not autoplay, please click to play or view the interactive demo page for the best experience.

---

## 2. BLURR in a Nutshell

Modern VLA controllers (Pi-0, OpenVLA, etc.) can solve diverse manipulation tasks,  
but their **inference stack is too heavy** for:

- responsive web demos, and  
- high-frequency robot control on commodity GPUs.

**BLURR** plugs into an existing controller **without retraining** and restructures the inference pathway:

1. **Reduce redundant prefix computation**  
   - Only compute instruction tokens once per episode.  
   - Reuse the instruction KV cache at every control step.

2. **Minimize per-step token cost**  
   - Shorter rollout horizon (fewer flow steps).  
   - Single-step control for SimplerEnv Bridge tasks.

3. **Maximize tensor-core utilization**  
   - BF16 execution, `torch.compile`, and FlashAttention in the decoder.  

At the end, we get **up to 9.5× lower latency**, **~0.5× peak VRAM**, and **9.2× higher effective GFLOPS**,  
while preserving state-of-the-art manipulation success.

---

## 3. Demo Tasks (SimplerEnv Bridge)

Our web demo exposes BLURR on four in-domain SimplerEnv Bridge tasks.  
Each clip below corresponds to a row in Figure 3 of the paper.

| Task name           | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| carrot-on-plate     | Move a carrot from the table surface to the target plate.   |
| eggplant-in-rack    | Insert an eggplant into a target slot of the rack.          |
| spoon-on-cloth      | Place the spoon stably on top of a folded cloth.            |
| block-stacking      | Stack blocks into a specified target configuration.         |

---

## 4. Efficiency Highlights

BLURR keeps all model weights unchanged and only modifies the inference stack.

### 3.1 Single-step efficiency (H100, 224×224 RGB, 256 tokens)

This table corresponds to Table 1 in the paper.

| Configuration        | Latency (ms) | VRAM (GB) | GFLOPS  |
| -------------------- | -----------: | --------: | ------: |
| OpenVLA              |       217.8  |    14.33  |  5,835  |
| OpenVLA-OFT          |        91.2  |    14.48  | 49,886  |
| Pi-0 baseline        |       111.6  |    13.58  | 39,038  |
| Interleave-Pi-0      |       162.1  |    13.61  |  7,989  |
| **BLURR-Pi-0 (ours)**|   **17.1**   | **7.20**  | **73,525** |

BLURR-Pi-0 roughly **doubles GFLOPS** over the Pi-0 baseline,  
while cutting both **latency** and **peak VRAM** by large margins.

### 3.2 Ablation: where do the gains come from?

This reproduces the impact of each BLURR component (Table 2 in the paper).

| Configuration                     | Latency (ms) | VRAM (GB) |
| --------------------------------- | -----------: | --------: |
| Interleave-Pi-0 (FP32, 10 steps)  |       162.1  |   13.61   |
| + BF16 only (10 steps)            |        88.2  |   13.58   |
| + `torch.compile` (10 steps)      |        56.7  |    6.15   |
| + fewer flow steps (6 steps)      |        44.7  |    7.28   |
| + fewer flow steps (4 steps)      |        34.8  |    7.29   |
| + KV cache                        |        31.9  |    7.32   |
| + FlashAttention                  |        27.4  |    7.30   |
| **Full BLURR (1 step)**          |   **17.1**   | **7.20**  |

---

## 5. Manipulation Performance (SimplerEnv)

Despite aggressive acceleration, BLURR maintains competitive success rates  
on all four Bridge tasks (Table 3 in the paper; 100 evaluation episodes per task).

| Model             | Carrot | Spoon | Blocks | Eggplant | Avg. |
| ----------------- | :----: | :---: | :----: | :------: | :--: |
| OpenVLA           | 0.47   | 0.44  | 0.63   | 0.68     | 0.56 |
| MiniVLA           | 0.42   | 0.67  | 0.69   | 0.18     | 0.49 |
| Baseline Pi-0     | 0.53   | 0.84  | 0.53   | 0.88     | 0.69 |
| Interleave-Pi-0   | 0.59   | 0.89  | 0.53   | 0.79     | 0.70 |
| **BLURR-Pi-0**    | 0.54   | 0.91  | 0.46   | 0.93     | **0.71** |

BLURR-Pi-0 matches or slightly improves over Interleave-Pi-0 on average,  
while being **much cheaper** to run.

---

## 6. Project Structure

```text
BLURR-A-Boosted-Low-Resource-Inference-for-Vision-Language-Action-Model/
├── README.md
├── demo/               # Web demo source code & assets
    ├── index.html
    ├── demo_carrot_on_plate_base.mp4
    ├── demo_carrot_on_plate.mp4
    └── ... (other clips)
```
