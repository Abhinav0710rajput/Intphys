import cv2
import torch
import os
from pathlib import Path

def process_video(video_path, target_fps=6, window_size=30, prediction=20, resize_hw=256):
    """
    Process videos: load, resize, change fps, and create sliding-window clips.
    
    Args:
        video_path: Path to directory containing mp4 videos or path to single video
        target_fps: Target frames per second (default: 6)
        window_size: Maximum window size (default: 30)
        prediction: Prediction/target frames (default: 20)
        resize_hw: Target height and width (default: 256)
    
    Returns:
        all_video_clips: List of lists, where each inner list contains tensor clips for one video
        video_names: List of video filenames
    """
    
    # Get all video paths
    if os.path.isdir(video_path):
        video_files = sorted([str(f) for f in Path(video_path).glob('*.mp4')])
    else:
        video_files = [video_path]
    
    video_names = [os.path.basename(vf) for vf in video_files]
    all_video_clips = []
    
    for vid_path in video_files:
        # Load video
        cap = cv2.VideoCapture(vid_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame skip for target fps
        frame_skip = int(original_fps / target_fps) if original_fps > target_fps else 1
        
        # Read and process all frames
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only keep frames according to target fps
            if frame_idx % frame_skip == 0:
                # Resize to (resize_hw, resize_hw)
                frame_resized = cv2.resize(frame, (resize_hw, resize_hw))
                # Convert BGR to RGB and normalize to [0, 1]
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            frame_idx += 1
        
        cap.release()
        
        # Convert to tensor: shape (T, H, W, C)
        frames_tensor = torch.tensor(frames, dtype=torch.float32) / 255.0
        total_frames = frames_tensor.shape[0]
        
        # Sliding window with stride 1.
        # Every window has exactly (window_size - prediction) context frames and
        # `prediction` target frames, so all windows are directly comparable.
        # The growing-context approach was removed because it produced empty context
        # masks for small window sizes (T <= tubelet_size) and dominated the loss
        # curves with incomparable high-error windows, hiding the physics signal.
        video_clips = []
        for start_idx in range(0, total_frames - window_size + 1):
            clip = frames_tensor[start_idx:start_idx + window_size]  # (window_size, H, W, C)
            video_clips.append(clip)
        
        all_video_clips.append(video_clips)
    
    return all_video_clips, video_names