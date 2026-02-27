import pickle
import numpy as np
import matplotlib.pyplot as plt
from process_video import *


with open("encoder_outputs.pkl", "rb") as f:
    enc = pickle.load(f)

with open("predictor_outputs.pkl", "rb") as f:
    pred = pickle.load(f)

with open("video_names.pkl", "rb") as f:
    names = pickle.load(f)

print(f"Number of videos: {len(names)}")
print(f"Number of encoder outputs: {len(enc)}")
print(f"Number of predictor outputs: {len(pred)}")

for i in range(len(names)):
    print(f"Video {i+1}: {names[i]}")
    print(f"  Encoder outputs windows: {len(enc[i])}")
    print(f"  Predictor outputs windows: {len(pred[i])}")
    loss = []
    for j in range(len(enc[i])):
        print(f"    Window {j+1}:")
        print(f"      Encoder output shape: {enc[i][j].shape}")
        print(f"      Predictor output shape: {pred[i][j].shape}")
        # print l2 norm between encoder and predictor outputs for the window

        l2_per_token = np.linalg.norm(enc[i][j] - pred[i][j], axis=-1)

        loss.append(l2_per_token.mean())

        print(f"      L2 norm between encoder and predictor outputs: {l2_per_token.mean()}")

    avg_loss = np.mean(loss)
    print(f"  Average L2 norm for video {names[i]}: {avg_loss}")
    max_loss = np.max(loss)
    print(f"  Max L2 norm for video {names[i]}: {max_loss}")

    
with open("per_token_losses.pkl", "rb") as f:
    losses = pickle.load(f)


plt.figure(figsize=(14, 6), constrained_layout=True)

for i in range(len(names)):
    window_losses = losses[i]
    all_per_token_losses = np.concatenate(window_losses, axis=0)

    mean_loss = all_per_token_losses.mean()
    std_loss = all_per_token_losses.std()

    print(f"Video {i+1}: {names[i]}")
    print(f"  Mean per-token loss: {mean_loss:.6f}")
    print(f"  Std per-token loss: {std_loss:.6f}")

    plt.plot(all_per_token_losses, label=names[i], linewidth=1.5)

plt.title("Per-Token Loss Curves for All Videos")
plt.xlabel("Token Index")
plt.ylabel("Loss")
plt.grid(True)

plt.legend()
plt.savefig("per_token_losses_.png", dpi=300)
plt.close()




"""
conda activate /scratch/$USER/envs/proj
"""