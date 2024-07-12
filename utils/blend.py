import sys
import argparse
import numpy as np
import os
import cv2
import glob
import dlib
import subprocess
import shutil
from tqdm import tqdm
from gfpgan import GFPGANer
# Add the parent directory to the Python path explicitly
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Now import from the utils module
from utils.common import get_versioned_filename

GFPGAN_arch = 'clean'
GFPGAN_channel_multiplier = 2
GFPGAN_model_name = 'GFPGANv1.4'
GFPGAN_url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
GFPGAN_model_path = os.path.join('experiments/pretrained_models', GFPGAN_model_name + '.pth')
restorer = GFPGANer(
        model_path=GFPGAN_model_path,
        upscale=2,
        arch=GFPGAN_arch,
        channel_multiplier=GFPGAN_channel_multiplier,
        bg_upsampler=None)

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")


def restore_face(img_path):
    # read image
    # print(img_path)
    input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # restore faces and background if necessary
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        input_img,
        has_aligned=True,
        only_center_face=True,
        paste_back=True,
        weight=0.5)
    # save restored img
    if restored_img is not None:
        cv2.imwrite(img_path,restored_img)

def Processmain(samelength_path, pre_blend_path):
    # Extract frames from the samelength video
    print('Extracting frames from samelength video')
    same_length_dir = os.path.join(os.path.dirname(samelength_path), 'samelength')
    if not os.path.exists(same_length_dir):
        os.makedirs(same_length_dir)
    extract_frames_from_video(samelength_path, same_length_dir)

    # Extract frames from the pre_blend video
    print('Extracting frames from pre_blend video')
    pre_blend_dir = os.path.join(os.path.dirname(pre_blend_path), 'pre_blend')
    if not os.path.exists(pre_blend_dir):
        os.makedirs(pre_blend_dir)
    extract_frames_from_video(pre_blend_path, pre_blend_dir)

    print('Tracking Face of Lip-synced (LipSick) video please wait..')

    # Call the blending function
    outputpath = blend_videos(same_length_dir, pre_blend_dir, samelength_path, pre_blend_path)
    return outputpath

def extract_frames_from_video(video_path, save_dir):
    videoCapture = cv2.VideoCapture(video_path)
    frames = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    old_frame = []
    for i in range(frames):
        ret, frame = videoCapture.read()
        if not ret:
            break
        result_path = os.path.join(save_dir, str(i).zfill(6) + '.jpg')
        old_frame.append(result_path)
        cv2.imwrite(result_path, frame)
    for perFrame in tqdm(old_frame, desc="Processing frames for Alpha"):
        restore_face(perFrame)
    return (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

def load_landmark_dlib(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    if not faces:
        raise ValueError("No faces found in the image.")
    shape = landmark_predictor(gray, faces[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    return landmarks

def alpha_blend_face(original_frame, generated_frame, landmarks):
    mask = np.zeros(original_frame.shape[:2], dtype=np.float32)
    points = cv2.convexHull(np.concatenate((landmarks[3:15], landmarks[30:36], landmarks[48:68])))

    if len(points) < 3:
        raise ValueError("Convex hull could not be formed. Not enough points.")

    cv2.fillConvexPoly(mask, points, 1.0)
    mask = cv2.GaussianBlur(mask, (51, 51), 30)
    mask = mask[..., np.newaxis]
    blended_frame = original_frame * (1 - mask) + generated_frame * mask

    return blended_frame.astype(np.uint8)

def blend_videos(same_length_dir, pre_blend_dir, samelength_path, pre_blend_path):
    # Get frames from both videos
    same_length_frame_path_list = glob.glob(os.path.join(same_length_dir, '*.jpg'))
    same_length_frame_path_list.sort()

    pre_blend_frame_path_list = glob.glob(os.path.join(pre_blend_dir, '*.jpg'))
    pre_blend_frame_path_list.sort()
    pre_blend_landmark_data = np.array([load_landmark_dlib(frame) for frame in pre_blend_frame_path_list])

    # Blend frames from pre_blend video onto samelength video using Alpha blending
    output_video_path = samelength_path.replace('samelength.mp4', '_lipsick_blend.mp4')
    output_video_path = get_versioned_filename(output_video_path)  # Ensure unique filename
    video_size = (int(cv2.VideoCapture(samelength_path).get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cv2.VideoCapture(samelength_path).get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videowriter = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, video_size)

    for i, (same_length_frame_path, pre_blend_frame_path, landmark_data) in enumerate(zip(same_length_frame_path_list, pre_blend_frame_path_list, pre_blend_landmark_data)):
        sys.stdout.write(f'\rAlpha blending frame {i+1}/{len(same_length_frame_path_list)}')
        sys.stdout.flush()  # Make sure to flush the output buffer

        same_length_frame = cv2.imread(same_length_frame_path)
        pre_blend_frame = cv2.imread(pre_blend_frame_path)

        try:
            blended_frame = alpha_blend_face(same_length_frame, pre_blend_frame, landmark_data)
        except ValueError as e:
            print(f"Skipping frame {i+1} due to error: {e}")
            blended_frame = same_length_frame  # Use original frame if blending fails

        videowriter.write(blended_frame)

    videowriter.release()

    # Store the output video path for later use
    output_video_path = output_video_path

    # Add audio to the blended video
    final_video_path = get_versioned_filename(output_video_path.replace('_lipsick_blend.mp4', 'LipSick_Blend.mp4'))

    cmd = f'ffmpeg -i "{output_video_path}" -i "{pre_blend_path}" -c:v libx264 -crf 23 -c:a aac -strict experimental "{final_video_path}"'
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Clean up intermediate files
    os.remove(output_video_path)
    os.remove(samelength_path)
    os.remove(pre_blend_path)
    shutil.rmtree(same_length_dir)
    shutil.rmtree(pre_blend_dir)
    return final_video_path

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Alpha blend two videos based on facial landmarks')
#     parser.add_argument('--samelength_video_path', type=str, required=True, help='Path to the samelength.mp4 video')
#     parser.add_argument('--pre_blend_video_path', type=str, required=True, help='Path to the pre_blend.mp4 video')
#     args = parser.parse_args()

#     main(args.samelength_video_path, args.pre_blend_video_path)
