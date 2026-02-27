import torch
import numpy as np
from process_video import *


def autoreg_mask(video_size,predict_frames,tubelet_size=2,patch_size=16,batch_size=1):
    T, H, W, C = video_size
    assert C == 3, "Expected RGB video with 3 channels."

    #### T = context + predict

    num_t = T // tubelet_size
    num_h = H // patch_size
    num_w = W // patch_size

    tokens_per_t = num_h * num_w

    total_tokens = num_t * tokens_per_t

    predicted_tubelets = predict_frames // tubelet_size
    assert predicted_tubelets > 0

    target_tokens = predicted_tubelets * tokens_per_t

    context_ids = torch.arange(0, total_tokens - target_tokens)
    target_ids  = torch.arange(total_tokens - target_tokens, total_tokens)

    # reshape to [B, N, 1]
    context_mask = context_ids.unsqueeze(0).repeat(batch_size, 1)#.unsqueeze(-1)
    target_mask  = target_ids.unsqueeze(0).repeat(batch_size, 1)#.unsqueeze(-1)

    return context_mask, target_mask



#testing

if __name__ == "__main__":
    video_size = (64, 256, 256, 3)
    predict_frames = 16
    tubelet_size = 2
    patch_size = 16
    batch_size = 1

    context_mask, target_mask = autoreg_mask(video_size, predict_frames, tubelet_size, patch_size, batch_size)
    