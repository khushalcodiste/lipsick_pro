from moviepy.editor import VideoFileClip, CompositeVideoClip, AudioFileClip, ImageSequenceClip
import os

def load_frames_from_folder(folder_path):
    frames = []
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])  # Load only PNG files to ensure transparency
    for file in files:
        frames.append(os.path.join(folder_path, file))
    return frames

def create_video_clip_from_frames(frames, fps):
    clip = ImageSequenceClip(frames, fps=fps, load_images=True)
    return clip

# Load the main background video
background_video = VideoFileClip("background.mp4")

# Load frames from the folder
frames_folder = "inference_result\\tempunet"
frames = load_frames_from_folder(frames_folder)

# Create a video clip from the frames
fps = 25  # Set the frame rate for your explanation video
explanation_video_clip = create_video_clip_from_frames(frames, fps)

# Load the separate audio file
audio_clip = AudioFileClip("speech.wav")

# Set the audio for the explanation video
explanation_video_clip = explanation_video_clip.set_audio(audio_clip)

# Resize the explanation video to fit the desired size
explanation_video_clip = explanation_video_clip.resize(height=background_video.h * 0.45)  # Resize to 25% of the background video's height

# Position the explanation video at the top-right corner
explanation_video_clip = explanation_video_clip.set_position(("right", "bottom"))

# Trim the background video to match the duration of the explanation video
background_video = background_video.subclip(0, explanation_video_clip.duration)

# Composite the two videos together
final_video = CompositeVideoClip([background_video, explanation_video_clip])

# Write the result to a file
final_video.write_videofile("final_video_test.mp4", codec="libx264")
