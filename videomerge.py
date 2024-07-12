import moviepy.editor as mp

def merge_videos_with_crossfade(video_paths, crossfade_duration, output_path):
    clips = [mp.VideoFileClip(path) for path in video_paths]    # Reading all the video path

    fps = clips[0].fps
    size = clips[0].size
    clips = [clip.set_fps(fps).resize(size) for clip in clips]    # Making common fps for all the videos

    # Merging two videos together
    def crossfade_transition(clip1, clip2, duration):
        return mp.CompositeVideoClip([
            clip1,
            clip2.set_start(clip1.duration - duration).crossfadein(duration)
        ], size=clip1.size).subclip(0, clip1.duration + clip2.duration - duration)


    final_clip = clips[0]         # Reading first clip

    # Iterating all videos present in a list
    for i in range(1, len(clips)):
        final_clip = crossfade_transition(final_clip, clips[i], crossfade_duration)

    final_clip.write_videofile(output_path, codec="libx264", fps=fps)    # Creating Output Video

# Example usage
video_paths = ["bg_1.mp4", "bg_2.mp4", "bg_3.mp4","bg_4.mp4","bg_5.mp4"]
crossfade_duration = 2   # HyperParameter for deciding amount of time through which transition take place(in seconds)
output_path = "finalmerged_op.mp4"

merge_videos_with_crossfade(video_paths, crossfade_duration, output_path)








