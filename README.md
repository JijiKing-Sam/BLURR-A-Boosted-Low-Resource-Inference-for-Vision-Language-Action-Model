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

---

## 1. BLURR in a Nutshell

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

## 2. Demo Tasks (SimplerEnv Bridge)

Our web demo exposes BLURR on four in-domain SimplerEnv Bridge tasks.  
Each clip below corresponds to a row in Figure 3 of the paper.

| Task name           | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| carrot-on-plate     | Move a carrot from the table surface to the target plate.   |
| eggplant-in-rack    | Insert an eggplant into a target slot of the rack.          |
| spoon-on-cloth      | Place the spoon stably on top of a folded cloth.            |
| block-stacking      | Stack blocks into a specified target configuration.         |

---

## 3. Efficiency Highlights

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

## 4. Manipulation Performance (SimplerEnv)

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

5. Demo Clips

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
<video src="demo/demo_carrot_on_plate_base.mp4" width="100%" autoplay loop muted playsinline></video>
</td>
<td>
<video src="demo/demo_carrot_on_plate.mp4" width="100%" autoplay loop muted playsinline></video>
</td>
</tr>

<!-- Task 2: Spoon -->

<tr>
<td colspan="2" align="center"><b>2. Spoon on Cloth</b></td>
</tr>
<tr>
<td>
<video src="demo/demo_spoon_on_cloth_base.mp4" width="100%" autoplay loop muted playsinline></video>
</td>
<td>
<video src="demo/demo_spoon_on_cloth.mp4" width="100%" autoplay loop muted playsinline></video>
</td>
</tr>

<!-- Task 3: Eggplant -->

<tr>
<td colspan="2" align="center"><b>3. Eggplant in Rack</b></td>
</tr>
<tr>
<td>
<video src="demo/demo_eggplant_in_rack_base.mp4" width="100%" autoplay loop muted playsinline></video>
</td>
<td>
<video src="demo/demo_eggplant_in_rack.mp4" width="100%" autoplay loop muted playsinline></video>
</td>
</tr>

<!-- Task 4: Blocks -->

<tr>
<td colspan="2" align="center"><b>4. Block Stacking</b></td>
</tr>
<tr>
<td>
<video src="demo/demo_block_stacking_base.mp4" width="100%" autoplay loop muted playsinline></video>
</td>
<td>
<video src="demo/demo_block_stacking.mp4" width="100%" autoplay loop muted playsinline></video>
</td>
</tr>
</table>

Note: If videos do not autoplay, please click to play or view the interactive demo page for the best experience.

---

## 6. Project Structure (minimal)

```text
BLURR-A-Boosted-Low-Resource-Inference-for-Vision-Language-Action-Model/
├── README.md
├── demo/               # Web demo source code & assets
    ├── index.html
    ├── demo_carrot_on_plate_base.mp4
    ├── demo_carrot_on_plate.mp4
    └── ... (other clips)


