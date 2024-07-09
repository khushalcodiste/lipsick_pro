# from moviepy.editor import VideoFileClip, CompositeVideoClip

# # Load the main background video
# background_video = VideoFileClip("background.mp4")

# # Load the second video (you explaining things)
# explanation_video = VideoFileClip("Nishant_org.mp4")

# # Resize the explanation video to fit the desired size
# explanation_video = explanation_video.resize(h)  # Resize to 25% of the background video's height

# # Position the explanation video at the top-right corner
# explanation_video = explanation_video.set_position(("right", "bottom"))

# # Composite the two videos together
# final_video = CompositeVideoClip([background_video, explanation_video])

# # Write the result to a file
# final_video.write_videofile("final_video.mp4", codec="libx264")

import cv2
import numpy as np
from moviepy.editor import VideoFileClip, CompositeVideoClip
from moviepy.video.VideoClip import ImageClip

def remove_background(frame, lower_bound, upper_bound):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask_inv = cv2.bitwise_not(mask)
    bg_removed = cv2.bitwise_and(frame, frame, mask=mask_inv)
    return bg_removed

def process_frame(frame):
    lower_bound = np.array([35, 100, 100])
    upper_bound = np.array([85, 255, 255])
    frame = remove_background(frame, lower_bound, upper_bound)
    return frame

def apply_chroma_key(video_path):
    video = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = process_frame(frame)
        frames.append(frame)
    
    video.release()
    return frames

def create_video_clip(frames, fps):
    height, width, layers = frames[0].shape
    size = (width, height)
    out = cv2.VideoWriter('temp_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    return VideoFileClip('temp_video.mp4')

# Load the main background video
background_video = VideoFileClip("background.mp4")

# Load and process the explanation video to remove the background
frames = apply_chroma_key("Nishant_org.mp4")
fps = int(cv2.VideoCapture("Nishant_org.mp4").get(cv2.CAP_PROP_FPS))
explanation_video = create_video_clip(frames, fps)

# Resize the explanation video to fit the desired size
explanation_video = explanation_video.resize(height=background_video.h * 0.25)  # Resize to 25% of the background video's height

# Position the explanation video at the top-right corner
explanation_video = explanation_video.set_position(("right", "top"))

# Composite the two videos together
final_video = CompositeVideoClip([background_video, explanation_video])

# Write the result to a file
final_video.write_videofile("final_video.mp4", codec="libx264")
