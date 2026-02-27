import cv2
import numpy as np
import os
import glob
import random

video_dir = "/home/ar10026/intphys/videos"
video_files = glob.glob(os.path.join(video_dir, "*.mp4"))

for video_path in video_files:
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    random.shuffle(frames)

    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(video_dir, f"{base}_shuffled.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

    print(f"Saved: {out_path}")


