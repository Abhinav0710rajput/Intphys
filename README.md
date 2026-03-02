# VJEPA2 Intuitive Physics Benchmark

> *Can a video foundation model be surprised by the impossible?*

---

## Overview

This project probes the **intuitive physics** capabilities of [V-JEPA 2](https://ai.meta.com/research/publications/v-jepa-2/) — Meta's state-of-the-art video Joint-Embedding Predictive Architecture — by measuring its **surprisal** (prediction error) on physically plausible versus physically implausible video events.

The central hypothesis is elegant: a model that has internalized the laws of physics should find realistic scenes *easy* to predict, and physically impossible ones *surprising*. By quantifying this gap in latent representation space, we ask whether VJEPA2 has learned an implicit world model — one that encodes gravity, object permanence, and causal dynamics — purely from video.

---

## The Core Idea

```
  Video frames  ──►  [V-JEPA2 Encoder]  ──►  latent tokens
                            │
                     context tokens
                            │
                            ▼
                    [V-JEPA2 Predictor]  ──►  predicted future tokens
                                                      │
                                              compare (MSE / L2)
                                                      │
                            ground-truth encoder      │
                            of future frames  ◄───────┘

  Surprisal = || predicted future  −  actual future ||²
```

**Low surprisal** → the model expected this. Physics makes sense.
**High surprisal** → the model was caught off-guard. Something impossible happened.

---

## Architecture

### `process_video.py` — Video Ingestion
Loads MP4 files from a directory, resamples to a target frame rate, resizes frames to 256×256, and slices the resulting tensor into a **sliding window** of fixed length. Each window contains a contiguous context + future segment, ensuring all windows are directly comparable.

### `autoreg_mask.py` — Autoregressive Masking
Generates **context** and **target** token masks for VJEPA2's spatiotemporal token space. Tokens are computed over 3D tubelets (patch_size=16, tubelet_size=2), so the mask indices correspond to the final `prediction_frames // tubelet_size` temporal groups — the frames the model must predict.

### `main.py` / `new.py` — Inference Pipeline
Runs the two-pass evaluation:
1. **Masked forward pass** — model sees only context tokens, predictor generates future token predictions.
2. **Full forward pass** (`skip_predictor=True`) — encoder embeds the *entire* video, yielding ground-truth latents for the future frames.

The MSE between these two is the per-window **surprisal score**.

### `analysis.py` / `plot.py` — Results & Visualization
Aggregates per-token and per-window losses across all videos. Produces:
- **Loss-over-time curves** — surprisal as a function of sliding-window position
- **Bar chart** — mean surprisal per video for direct comparison

---

## Model

| Component | Details |
|-----------|---------|
| Model | `facebook/vjepa2-vitl-fpc64-256` |
| Architecture | ViT-L encoder + JEPA predictor |
| Input resolution | 256 × 256 |
| Dtype | `float16` |
| Attention | Scaled Dot-Product Attention (SDPA) |

---

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `window_size` | 48 | Total frames per sliding window |
| `prediction` | 2 | Target (future) frames per window |
| `target_fps` | 6 | Resampled video frame rate |
| `resize_hw` | 256 | Spatial resolution |
| `tubelet_size` | 2 | Temporal depth of each token |
| `patch_size` | 16 | Spatial patch size |

---

## Setup

```bash
# Create and activate the conda environment
conda create -n proj python=3.10
conda activate /scratch/$USER/envs/proj

# Install dependencies
pip install torch transformers opencv-python numpy matplotlib
```

---

## Usage

### 1. Prepare Videos

Place your `.mp4` files (realistic and physically implausible) into the `videos/` directory.

### 2. Run Inference

```bash
python new.py
```

This processes all videos, runs the two-pass encoder/predictor comparison, and saves:

```
encoder_outputs_window_48_pred_2_fps_6.pkl
predictor_outputs_window_48_pred_2_fps_6.pkl
l2_losses_window_48_pred_2_fps_6.pkl
per_token_losses_window_48_pred_2_fps_6.pkl
video_names.pkl
```

### 3. Visualize Results

```bash
python plot.py
```

Generates a dual-panel figure: surprisal curves over time (top) and mean surprisal bar chart per video (bottom).

### 4. Run on HPC (SLURM)

```bash
sbatch run_main.sbatch
```

Requests 1× L40S GPU, 16 GB RAM, 4 CPUs, 4-hour wall time.

---

## Results Interpretation

A model with genuine intuitive physics should show a clear separation:

```
  Realistic video    ──►  Low surprisal   (model predicted it well)
  Impossible video   ──►  High surprisal  (model was surprised)
```

The mean surprisal bar chart provides a direct probe of this discriminative capacity. A significant gap between physically plausible and implausible categories is evidence that VJEPA2 has internalized a structured model of the physical world — not just statistical video priors, but something closer to causal, physical understanding.

---

## File Structure

```
intphys/
├── videos/                  # Input MP4 videos (realistic + impossible events)
├── main.py                  # Original inference script
├── new.py                   # Refined two-pass inference pipeline
├── process_video.py         # Video loading, FPS resampling, sliding windows
├── autoreg_mask.py          # Context/target token mask generation
├── analysis.py              # Loss analysis and statistics
├── plot.py                  # Surprisal visualization (curves + bar chart)
├── video_edit.py            # Video preprocessing utilities
├── run_main.sbatch          # SLURM job script for HPC cluster
└── *.pkl                    # Saved inference outputs
```

---

## Background

**Intuitive physics** is a core component of human cognition — infants as young as a few months old show surprise (measured by looking time) when objects appear to violate basic physical laws such as solidity, continuity, or gravity.

**V-JEPA 2** is a self-supervised video model trained with a joint-embedding objective: instead of predicting raw pixels, it predicts abstract latent representations of future video content. This makes it a natural candidate for probing physical understanding, since its internal representations must capture the *structure* of visual dynamics.

This benchmark draws inspiration from the [IntPhys](https://intphys.cognitive-ml.fr/) and [ADEPT](https://openreview.net/forum?id=Bygg2yBtvH) evaluation frameworks, adapting the surprisal-based methodology to the latent prediction space of a modern video foundation model.

---

## References

- Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, Meta AI Research, 2025.
- Riochet et al., *IntPhys: A Benchmark and a Model for Physical Intuition in Visual Agents*, ICLR 2019.
- Smith et al., *ADEPT: A Continuous-Time Model of the Physical World*, NeurIPS 2019.
