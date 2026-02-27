"""
conda activate /scratch/$USER/envs/proj
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

processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
model = AutoModel.from_pretrained(
    "facebook/vjepa2-vitl-fpc64-256",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

device = model.device

window_size = 48
prediction = 34
hw = 256
#video_size = (window_size, hw, hw, 3)
video_path = "videos"
# video = nan # add code to load video 
videos, names = process_video(video_path, target_fps=6, window_size=window_size, prediction=prediction, resize_hw=256)
#just sample last window for first clip

encoder_outputs_all = []
predictor_outputs_all = []

i = 0
for video in videos:
    window_enc_out = []
    window_pred_out = []

    for window in video:
        context_mask, target_mask = autoreg_mask(window.shape, prediction)
        target_mask = target_mask.to(device)
        context_mask = context_mask.to(device)

        inputs = processor(window, return_tensors="pt")
        inputs = inputs.to(model.device, dtype=torch.float16)

        torch.cuda.empty_cache()

        outputs = model(
            **inputs,
            context_mask=[context_mask],
            target_mask=[target_mask]
        )

        # # Encoder output
        encoder_outputs = outputs.last_hidden_state
        # # Predictor output
        predictor_outputs = outputs.predictor_output.last_hidden_state

        ## GT comparison
        Nt = predictor_outputs.shape[1]
        encoder_target_output = encoder_outputs[:, -Nt:, :]

        window_enc_out.append(encoder_target_output.detach().cpu().numpy())
        window_pred_out.append(predictor_outputs.detach().cpu().numpy())
    
    encoder_outputs_all.append(window_enc_out)
    predictor_outputs_all.append(window_pred_out)
    print(f"Processed video {i+1}/{len(videos)}: {names[i]}")
    i += 1


with open("encoder_outputs.pkl", "wb") as f:
    pickle.dump(encoder_outputs_all, f)

with open("predictor_outputs.pkl", "wb") as f:
    pickle.dump(predictor_outputs_all, f)

with open("video_names.pkl", "wb") as f:
    pickle.dump(names, f)


# with open("list.pkl", "rb") as f:
#     lst = pickle.load(f)


        









# video = videos[0][-1]  # (48 or 35 most likely, 256, 256, 3)

# print("Video shape:", video.shape)

# context_mask, target_mask = autoreg_mask(video.shape, prediction)
# context_mask = context_mask.to(device)
# target_mask = target_mask.to(device)

# print(context_mask)


# inputs = processor(video, return_tensors="pt")
# inputs = inputs.to(model.device, dtype=torch.float16)

# torch.cuda.empty_cache()


# outputs = model(
#     **inputs,
#     context_mask=[context_mask],
#     target_mask=[target_mask]
# )


# # # Encoder output
# encoder_outputs = outputs.last_hidden_state

# # # Predictor output
# predictor_outputs = outputs.predictor_output.last_hidden_state


# ## GT comparison
# Nt = predictor_outputs.shape[1]
# encoder_target_output = encoder_outputs[:, -Nt:, :]




# print("Encoder target output:", encoder_target_output.shape)
# print("Predictor target output:", predictor_outputs.shape)

