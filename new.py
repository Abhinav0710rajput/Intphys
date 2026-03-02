"""
conda activate /scratch/$USER/envs/proj

Correct implementation for calculating L2 loss between encoder and predictor embeddings
for realistic vs unrealistic video comparison.
"""

import torch
import numpy as np 
from transformers import AutoModel, AutoVideoProcessor
from autoreg_mask import *
import pickle
from process_video import *



print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda version:", torch.version.cuda)

# Load model and processor
processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
model = AutoModel.from_pretrained(
    "facebook/vjepa2-vitl-fpc64-256",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

device = model.device
model.eval()  # Set to evaluation mode

# Video processing parameters
window_size = 48
prediction = 2
hw = 256
video_path = "videos"
target_fps = 6

# Process videos
videos, names = process_video(
    video_path, 
    window_size=window_size, 
    prediction=prediction, 
    target_fps=target_fps,
    resize_hw=256
)

# Storage for results
encoder_outputs_all = []
predictor_outputs_all = []
l2_losses_all = []
per_token_losses_all = []

print(f"\nProcessing {len(videos)} videos...")
print("=" * 60)

for i, video in enumerate(videos):
    window_enc_out = []
    window_pred_out = []
    window_l2_losses = []
    window_per_token_losses = []

    for j, window in enumerate(video):
        # Generate masks
        video_shape = window.shape  # (T, H, W, C)
        context_mask, target_mask = autoreg_mask(video_shape, prediction)
        target_mask = target_mask.to(device)
        context_mask = context_mask.to(device)

        # Process video
        inputs = processor(window, return_tensors="pt")
        inputs = inputs.to(device, dtype=torch.float16)

        torch.cuda.empty_cache()

        with torch.no_grad():
            # Step 1: Get predictor outputs (predictions based on context)
            masked_outputs = model(
                **inputs,
                context_mask=[context_mask],
                target_mask=[target_mask]
            )
            predictor_outputs = masked_outputs.predictor_output.last_hidden_state
            
            # Step 2: Get ground truth encoder outputs (WITHOUT masking)
            # This gives us the true embeddings for the target frames
            full_outputs = model(
                **inputs,
                skip_predictor=True  # Only run encoder, skip predictor
            )
            encoder_full_outputs = full_outputs.last_hidden_state
            
            # Step 3: Extract the target tokens using target_mask indices
            # The target_mask contains the indices of tokens we want to predict
            Nt = predictor_outputs.shape[1]
            
            # Method 1: If target tokens are the last Nt tokens (temporal autoregressive)
            # encoder_target_output = encoder_full_outputs[:, -Nt:, :]
            
            # Method 2: If you want to use exact indices from target_mask (more general)
            target_indices = target_mask[0].long()  # Shape: [Nt]
            encoder_target_output = encoder_full_outputs[:, target_indices, :]
            
            # Verify shapes match
            assert predictor_outputs.shape == encoder_target_output.shape, \
                f"Shape mismatch: predictor {predictor_outputs.shape} vs encoder {encoder_target_output.shape}"
            
            # Calculate L2 loss (MSE) - overall average
            l2_loss = torch.nn.functional.mse_loss(
                predictor_outputs.float(), 
                encoder_target_output.float(),
                reduction='mean'
            )
            
            # Calculate per-token L2 loss for detailed analysis
            # Shape: [batch_size (=1), num_tokens]
            per_token_loss = torch.mean(
                (predictor_outputs.float() - encoder_target_output.float()) ** 2, 
                dim=-1  # Average over embedding dimension
            )
            
            # Store results
            window_enc_out.append(encoder_target_output.detach().cpu().numpy())
            window_pred_out.append(predictor_outputs.detach().cpu().numpy())
            window_l2_losses.append(l2_loss.item())
            window_per_token_losses.append(per_token_loss.detach().cpu().numpy())
        
        # if j == 0:  # Print details for first window only
        #     print(f"  Window {j+1}: L2 Loss = {l2_loss.item():.6f}")
        #     print(f"    Encoder output shape: {encoder_full_outputs.shape}")
        #     print(f"    Predictor output shape: {predictor_outputs.shape}")
        #     print(f"    Target tokens shape: {encoder_target_output.shape}")
        #     print(f"    Per-token loss range: [{per_token_loss.min().item():.6f}, {per_token_loss.max().item():.6f}]")
        print(j+1)
    
    # Store all windows for this video
    encoder_outputs_all.append(window_enc_out)
    predictor_outputs_all.append(window_pred_out)
    l2_losses_all.append(window_l2_losses)
    per_token_losses_all.append(window_per_token_losses)
    
    # Print summary for this video
    avg_loss = np.mean(window_l2_losses)
    std_loss = np.std(window_l2_losses)

    #average per-token loss across all windows
    all_per_token_losses = np.concatenate(window_per_token_losses, axis=0)
    print(f"\nPer-token loss for video {names[i]}: mean = {all_per_token_losses.mean():.6f}, std = {all_per_token_losses.std():.6f}")
    print(f"\nVideo {i+1}/{len(videos)}: {names[i]}")
    print(f"  Average L2 Loss: {avg_loss:.6f} ± {std_loss:.6f}")
    print(f"  Number of windows: {len(window_l2_losses)}")
    print("-" * 60)

# Print final summary statistics
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

for i, (name, losses) in enumerate(zip(names, l2_losses_all)):
    avg_loss = np.mean(losses)
    std_loss = np.std(losses)
    min_loss = np.min(losses)
    max_loss = np.max(losses)
    print(f"{i+1}. {name}:")
    print(f"   Mean: {avg_loss:.6f} ± {std_loss:.6f}")
    print(f"   Range: [{min_loss:.6f}, {max_loss:.6f}]")

t = target_fps
if t is None:
    t = 0 #original fps

# Save results
print("\nSaving results...")
with open(f"encoder_outputs_window_{window_size}_pred_{prediction}_fps_{t}.pkl", "wb") as f:
    pickle.dump(encoder_outputs_all, f)

with open(f"predictor_outputs_window_{window_size}_pred_{prediction}_fps_{t}.pkl", "wb") as f:
    pickle.dump(predictor_outputs_all, f)

with open(f"l2_losses_window_{window_size}_pred_{prediction}_fps_{t}.pkl", "wb") as f:
    pickle.dump(l2_losses_all, f)

with open(f"per_token_losses_window_{window_size}_pred_{prediction}_fps_{t}.pkl", "wb") as f:
    pickle.dump(per_token_losses_all, f)

with open("video_names.pkl", "wb") as f:
    pickle.dump(names, f)

print("Done! Results saved to:")
print("  - encoder_outputs.pkl")
print("  - predictor_outputs.pkl")
print("  - l2_losses.pkl")
print("  - per_token_losses.pkl")
print("  - video_names.pkl")

# Optional: Quick comparison if you know which videos are realistic vs unrealistic
# Uncomment and modify indices as needed
"""
realistic_indices = [0, 1, 2]  # Example: first 3 videos are realistic
unrealistic_indices = [3, 4, 5]  # Example: last 3 videos are unrealistic

realistic_losses = [l2_losses_all[i] for i in realistic_indices]
unrealistic_losses = [l2_losses_all[i] for i in unrealistic_indices]

realistic_avg = np.mean([np.mean(losses) for losses in realistic_losses])
unrealistic_avg = np.mean([np.mean(losses) for losses in unrealistic_losses])

print("\n" + "=" * 60)
print("REALISTIC vs UNREALISTIC COMPARISON")
print("=" * 60)
print(f"Realistic videos average L2 loss: {realistic_avg:.6f}")
print(f"Unrealistic videos average L2 loss: {unrealistic_avg:.6f}")
print(f"Ratio (unrealistic/realistic): {unrealistic_avg/realistic_avg:.2f}x")
"""



import pickle
import numpy as np
import matplotlib.pyplot as plt


with open("video_names.pkl", "rb") as f:
    names = pickle.load(f)

with open(f"per_token_losses_window_{window_size}_pred_{prediction}_fps_{t}.pkl", "rb") as f:
    losses = pickle.load(f)

# ── unique colour per video ─────────────────────────────────────────────────
n_videos = len(names)
cmap = plt.get_cmap("tab20") if n_videos <= 20 else plt.get_cmap("hsv")
colors = [cmap(i / max(n_videos, 1)) for i in range(n_videos)]

fig, (ax_curves, ax_bar) = plt.subplots(
    2, 1, figsize=(14, 10),
    gridspec_kw={"height_ratios": [3, 1]},
)

video_means = []
for i in range(n_videos):
    window_losses = losses[i]

    # Scalar surprisal per window: mean MSE across all target tokens
    # per_token_loss stored as numpy shape [1, N_tokens] per window
    window_level_losses = [float(np.mean(w)) for w in window_losses]

    mean_loss = float(np.mean(window_level_losses))
    std_loss  = float(np.std(window_level_losses))
    video_means.append(mean_loss)

    print(f"Video {i+1}: {names[i]}")
    print(f"  Windows : {len(window_level_losses)}")
    print(f"  Mean loss: {mean_loss:.6f} ± {std_loss:.6f}")

    label = names[i].replace(".mp4", "").replace("_", " ")
    ax_curves.plot(window_level_losses, label=label,
                   linewidth=1.8, color=colors[i])

ax_curves.set_title("Surprisal (prediction error) per sliding window — higher = more surprised")
ax_curves.set_xlabel(f"Window index  (window size={window_size}, target={prediction} frames, fps={t})")
ax_curves.set_ylabel("Mean MSE over target tokens")
ax_curves.grid(True, alpha=0.4)
ax_curves.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

# ── bar chart: overall mean surprisal per video ──────────────────────────────
bar_labels = [n.replace(".mp4", "").replace("_", "\n") for n in names]
ax_bar.bar(range(n_videos), video_means, color=colors, edgecolor="k", linewidth=0.6)
ax_bar.set_xticks(range(len(names)))
ax_bar.set_xticklabels(bar_labels, fontsize=8)
ax_bar.set_ylabel("Mean surprisal")
ax_bar.set_title("Overall mean surprisal per video  (weird/noisy should be higher)")
ax_bar.grid(axis="y", alpha=0.4)

plt.tight_layout()
plt.savefig(f"per_token_losses_window_{window_size}_pred_{prediction}_fps_{t}.png",
            dpi=300, bbox_inches="tight")
plt.close()
print(f"\nPlot saved: per_token_losses_window_{window_size}_pred_{prediction}_fps_{t}.png")











































































































































































































































































