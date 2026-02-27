import pickle
import numpy as np
import matplotlib.pyplot as plt
from process_video import *



with open("video_names.pkl", "rb") as f:
    names = pickle.load(f)

with open("per_token_losses.pkl", "rb") as f:
    losses = pickle.load(f)

plt.figure(figsize=(16, 8))  # bigger figure

for i in range(len(names)):
    window_losses = losses[i]
    all_per_token_losses = np.concatenate(window_losses, axis=0)

    print(all_per_token_losses.shape)

    window_level_losses = [] 
    for w  in window_losses:
        window_level_losses.append(np.mean(w, axis=-1))

    mean_loss = np.array(window_level_losses).mean()
    std_loss = np.array(window_level_losses).std()

    print(f"Video {i+1}: {names[i]}")
    print(f"  Mean per-token loss: {mean_loss:.6f}")
    print(f"  Std per-token loss: {std_loss:.6f}")

    plt.plot(window_level_losses, label=names[i], linewidth=1.8)

plt.title("Per-Token Loss Curves for All Videos")
plt.xlabel("Frame Index")
plt.ylabel("Loss")
plt.grid(True)

# move legend OUTSIDE so it never interferes with the plot area
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

plt.savefig("per_token_losses.png", dpi=300, bbox_inches="tight")
plt.close()

"""
new and plot
"""