"""
Sweep over context lengths [4, 6, 8, 10, 12, 14] x fps [6, original] = 12 cases.

Loss follows the IntPhys2 / V-JEPA2 paper:
  - Target encoder sees the FULL unmasked clip
  - Targets are layer-normalised before loss
  - Loss = mean L1 between predictor output and normalised target tokens

Run:
    conda activate torchenv
    python sweep.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoVideoProcessor
from autoreg_mask import autoreg_mask
from process_video import process_video

# ── Config ────────────────────────────────────────────────────────────────────
WINDOW_SIZE     = 48
CONTEXT_LENGTHS = [4, 6, 8, 10, 12, 14]   # frames used as context
FPS_VALUES      = [6, None]                # None = original fps
VIDEO_PATH      = "videos"
RESULTS_DIR     = "."                      # where to write pkl / png files

# ── Load model once ───────────────────────────────────────────────────────────
print("Loading V-JEPA2 model...")
processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
model = AutoModel.from_pretrained(
    "facebook/vjepa2-vitl-fpc64-256",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa",
)
model.eval()
device = model.device
print(f"Model loaded on {device}\n")


def run_inference(videos, names, context, device):
    """Return per_token_losses_all: list[video] of list[window] of np.ndarray [1, N_target]."""
    prediction = WINDOW_SIZE - context
    per_token_losses_all = []

    for i, video in enumerate(videos):
        window_losses = []

        for j, window in enumerate(video):
            c_mask, t_mask = autoreg_mask(window.shape, prediction)
            t_mask = t_mask.to(device)
            c_mask = c_mask.to(device)

            inputs = processor(window, return_tensors="pt")
            inputs = inputs.to(device, dtype=torch.float16)
            torch.cuda.empty_cache()

            with torch.no_grad():
                # Predictor: context → predicted target tokens
                masked_out = model(**inputs, context_mask=[c_mask], target_mask=[t_mask])
                preds = masked_out.predictor_output.last_hidden_state.float()

                # Ground truth: full encoder on unmasked clip, then select target tokens
                full_out = model(**inputs, skip_predictor=True)
                enc_full = full_out.last_hidden_state.float()

                target_idx = t_mask[0].long()
                targets = enc_full[:, target_idx, :]

                # Layer-normalise targets (paper: only targets, not preds)
                targets = F.layer_norm(targets, (targets.shape[-1],))

                # Per-token L1 loss  →  shape [1, N_target]
                per_token = torch.mean(torch.abs(preds - targets), dim=-1)
                window_losses.append(per_token.cpu().numpy())

            win_mean = float(np.mean(window_losses[-1]))
            print(f"    [{names[i]}] win {j+1}/{len(video)}: L1={win_mean:.4f}")

        per_token_losses_all.append(window_losses)
        vid_mean = float(np.mean([np.mean(w) for w in window_losses]))
        print(f"  >> Video {i+1}/{len(videos)} {names[i]}  mean={vid_mean:.4f}")

    return per_token_losses_all


# ── Main sweep ────────────────────────────────────────────────────────────────
all_results = {}   # key: (context, fps_tag) → {'losses': ..., 'names': ...}

for target_fps in FPS_VALUES:
    fps_tag = target_fps if target_fps is not None else "orig"
    print(f"\n{'='*60}")
    print(f"  FPS = {fps_tag}")
    print(f"{'='*60}")

    videos, names = process_video(VIDEO_PATH, window_size=WINDOW_SIZE,
                                  target_fps=target_fps, resize_hw=256)
    print(f"Videos: {names}")
    print(f"Windows per video: {[len(v) for v in videos]}\n")

    # Save canonical name list once
    with open(os.path.join(RESULTS_DIR, "video_names.pkl"), "wb") as f:
        pickle.dump(names, f)

    for context in CONTEXT_LENGTHS:
        prediction = WINDOW_SIZE - context
        print(f"\n  Context={context}  Prediction={prediction}")

        losses = run_inference(videos, names, context, device)

        key = (context, fps_tag)
        all_results[key] = {"losses": losses, "names": names}

        t_label = target_fps if target_fps is not None else 0
        fname = os.path.join(
            RESULTS_DIR,
            f"per_token_losses_window_{WINDOW_SIZE}_ctx_{context}_fps_{t_label}.pkl",
        )
        with open(fname, "wb") as f:
            pickle.dump(losses, f)
        print(f"  Saved {fname}")

print("\nAll inference done. Generating plots...")


# ── Plotting ──────────────────────────────────────────────────────────────────
def mean_surprisal(losses_for_video):
    """Return scalar: mean L1 across all windows and tokens for one video."""
    return float(np.mean([np.mean(w) for w in losses_for_video]))


def is_shuffled(name):
    return "shuffled" in name.lower()


# ── Figure 1: 2×6 grid of bar charts (one per case) ──────────────────────────
n_ctx = len(CONTEXT_LENGTHS)
n_fps = len(FPS_VALUES)
fig1, axes = plt.subplots(
    n_fps, n_ctx,
    figsize=(4 * n_ctx, 5 * n_fps),
    sharey=False,
)
fig1.suptitle("Surprisal (L1 loss, layer-normed targets) per video\n"
              "— shuffled should be higher", fontsize=14, y=1.01)

for r, target_fps in enumerate(FPS_VALUES):
    fps_tag = target_fps if target_fps is not None else "orig"
    for c, context in enumerate(CONTEXT_LENGTHS):
        ax = axes[r][c]
        key = (context, fps_tag)
        if key not in all_results:
            ax.set_visible(False)
            continue

        data = all_results[key]
        names = data["names"]
        losses = data["losses"]

        means = [mean_surprisal(losses[i]) for i in range(len(names))]
        colors = ["#d62728" if is_shuffled(n) else "#1f77b4" for n in names]
        short_names = [n.replace(".mp4", "").replace("_", "\n") for n in names]

        bars = ax.bar(range(len(names)), means, color=colors, edgecolor="k",
                      linewidth=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(short_names, fontsize=6)
        ax.set_title(f"ctx={context} fps={fps_tag}", fontsize=9)
        ax.set_ylabel("Mean L1" if c == 0 else "")
        ax.grid(axis="y", alpha=0.3)

        # Annotate bars with value
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=5)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#1f77b4", label="Normal"),
                   Patch(facecolor="#d62728", label="Shuffled")]
fig1.legend(handles=legend_elements, loc="upper right", fontsize=10)

plt.tight_layout()
fig1_path = os.path.join(RESULTS_DIR, "sweep_grid_barcharts.png")
fig1.savefig(fig1_path, dpi=200, bbox_inches="tight")
plt.close(fig1)
print(f"Saved {fig1_path}")


# ── Figure 2: Shuffled − Normal gap vs context length (one panel per fps) ────
# Pair videos by base name: "weird_1_.mp4" ↔ "weird_1__shuffled.mp4"
def find_pairs(names):
    """Return list of (normal_idx, shuffled_idx, base_label) tuples."""
    pairs = []
    for i, name in enumerate(names):
        if not is_shuffled(name):
            # find matching shuffled
            base = name.replace(".mp4", "")
            for j, sname in enumerate(names):
                if is_shuffled(sname) and base in sname:
                    pairs.append((i, j, base.replace("_", " ").strip()))
                    break
    return pairs


fig2, axes2 = plt.subplots(1, n_fps, figsize=(7 * n_fps, 6), sharey=True)
if n_fps == 1:
    axes2 = [axes2]

pair_colors = plt.cm.tab10(np.linspace(0, 1, 4))

for col, target_fps in enumerate(FPS_VALUES):
    fps_tag = target_fps if target_fps is not None else "orig"
    ax = axes2[col]

    # Need names from any key with this fps
    sample_key = (CONTEXT_LENGTHS[0], fps_tag)
    if sample_key not in all_results:
        continue
    names = all_results[sample_key]["names"]
    pairs = find_pairs(names)

    for pi, (ni, si, label) in enumerate(pairs):
        gaps = []
        for context in CONTEXT_LENGTHS:
            key = (context, fps_tag)
            if key not in all_results:
                gaps.append(np.nan)
                continue
            losses = all_results[key]["losses"]
            normal_mean = mean_surprisal(losses[ni])
            shuffled_mean = mean_surprisal(losses[si])
            gaps.append(shuffled_mean - normal_mean)

        ax.plot(CONTEXT_LENGTHS, gaps, marker="o", linewidth=2,
                color=pair_colors[pi], label=label)

    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_title(f"fps={fps_tag}", fontsize=12)
    ax.set_xlabel("Context frames", fontsize=11)
    if col == 0:
        ax.set_ylabel("Surprisal gap  (shuffled − normal)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

fig2.suptitle("Surprisal gap: shuffled vs normal\n"
              "Positive = model is more surprised by shuffled (expected)", fontsize=13)
plt.tight_layout()
fig2_path = os.path.join(RESULTS_DIR, "sweep_gap_vs_context.png")
fig2.savefig(fig2_path, dpi=200, bbox_inches="tight")
plt.close(fig2)
print(f"Saved {fig2_path}")


# ── Figure 3: Window-level loss curves for each fps, overlaid by context ─────
for target_fps in FPS_VALUES:
    fps_tag = target_fps if target_fps is not None else "orig"
    sample_key = (CONTEXT_LENGTHS[0], fps_tag)
    if sample_key not in all_results:
        continue
    names = all_results[sample_key]["names"]
    n_vid = len(names)

    fig3, axes3 = plt.subplots(n_vid, 1, figsize=(14, 3 * n_vid), sharex=False)
    if n_vid == 1:
        axes3 = [axes3]

    ctx_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(CONTEXT_LENGTHS)))

    for vi, vname in enumerate(names):
        ax = axes3[vi]
        for ci, context in enumerate(CONTEXT_LENGTHS):
            key = (context, fps_tag)
            if key not in all_results:
                continue
            w_losses = [float(np.mean(w)) for w in all_results[key]["losses"][vi]]
            ax.plot(w_losses, color=ctx_colors[ci], linewidth=1.5,
                    label=f"ctx={context}")
        label = vname.replace(".mp4", "")
        style = "--" if is_shuffled(vname) else "-"
        ax.set_title(f"{label}  ({'shuffled' if is_shuffled(vname) else 'normal'})",
                     fontsize=9)
        ax.set_ylabel("Mean L1", fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right", ncol=3)

    axes3[-1].set_xlabel("Window index", fontsize=10)
    fig3.suptitle(f"Window-level surprisal curves  (fps={fps_tag})\n"
                  f"Each line = different context length; dashed title = shuffled",
                  fontsize=12)
    plt.tight_layout()
    fig3_path = os.path.join(RESULTS_DIR, f"sweep_curves_fps_{fps_tag}.png")
    fig3.savefig(fig3_path, dpi=200, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved {fig3_path}")

print("\nAll done.")
