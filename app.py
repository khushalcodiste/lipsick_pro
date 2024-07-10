# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import FileResponse
import shutil
import os
from basicsr.utils.download_util import save_response_content
import numpy as np
import glob
import sys
import os
import cv2
import torch
import subprocess
import random
from collections import OrderedDict
import dlib
import shutil
import warnings
import tensorflow as tf
import argparse
import cv2
import glob
import numpy as np
import os
import torch
from multiprocessing import Pool, cpu_count
from basicsr.utils import imwrite
from tqdm import tqdm
from utils.blend import Processmain
# from gfpgan import GFPGANer


warnings.filterwarnings("ignore", category=UserWarning, message="Default grid_sample and affine_grid behavior has changed*")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from utils.deep_speech import DeepSpeech
from utils.data_processing import compute_crop_radius
from config.config import LipSickInferenceOptions
from models.LipSick import LipSick  # Import the LipSick model


#lipsick

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")
deepspeech_model_path = './asserts/output_graph.pb'
pretrained_lipsick_path = './asserts/pretrained_lipsick.pth'
auto_mask = True
DSModel = DeepSpeech(deepspeech_model_path)

#GFPGANv1.4
# GFPGAN_arch = 'clean'
# GFPGAN_channel_multiplier = 2
# GFPGAN_model_name = 'GFPGANv1.4'
# GFPGAN_url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
# GFPGAN_model_path = os.path.join('experiments/pretrained_models', GFPGAN_model_name + '.pth')
# restorer = GFPGANer(
#         model_path=GFPGAN_model_path,
#         upscale=2,
#         arch=GFPGAN_arch,
#         channel_multiplier=GFPGAN_channel_multiplier,
#         bg_upsampler=None)



model = LipSick(source_channel=3, ref_channel=15, audio_channel=29).to('cuda')
if not os.path.exists(pretrained_lipsick_path):
    raise Exception(f'Wrong path of pretrained model weight: {pretrained_lipsick_path}')
state_dict = torch.load(pretrained_lipsick_path, map_location=torch.device('cpu'))['state_dict']['net_g']
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] 
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.eval()



# app = FastAPI()

def get_versioned_filename(filepath):
    base, ext = os.path.splitext(filepath)
    counter = 1
    while os.path.exists(filepath):
        filepath = f"{base}({counter}){ext}"
        counter += 1
    return filepath

def convert_audio_to_wav(audio_path):
    output_path = os.path.splitext(audio_path)[0] + '.wav'
    if not audio_path.lower().endswith('.wav'):
        command = f'ffmpeg -i "{audio_path}" -acodec pcm_s16le -ar 16000 -ac 1 "{output_path}"'
        subprocess.run(command, shell=True, check=True)
    return output_path

# def restore_face(img_path):
#     # read image
#     img_name = os.path.basename(img_path)
#     basename, ext = os.path.splitext(img_name)
#     input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

#     # restore faces and background if necessary
#     cropped_faces, restored_faces, restored_img = restorer.enhance(
#         input_img,
#         has_aligned=True,
#         only_center_face=True,
#         paste_back=True,
#         weight=0.5)
#     # save restored img
#     if restored_img is not None:
#         extension = ext[1:]
#         # save_restore_path = os.path.join(output, f'{basename}.{extension}')
#         save_restore_path = img_path
#         cv2.imwrite(restored_img, save_restore_path)



def extract_frames_from_video(video_path, save_dir):
    video_capture = cv2.VideoCapture(video_path)
    frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    old_frame = []
    for i in range(frames):
        ret, frame = video_capture.read()
        if not ret:
            break
        result_path = os.path.join(save_dir, str(i).zfill(6) + '.jpg')
        old_frame.append(result_path)
        cv2.imwrite(result_path, frame)
    # for perFrame in tqdm(old_frame, desc="Processing frames"):
    #     restore_face(perFrame)
    # print("Face Restored")
    return (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

def load_landmark_dlib(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    if not faces:
        raise ValueError("No faces found in the image.")
    shape = landmark_predictor(gray, faces[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    return landmarks

def parse_reference_indices(indices_str):
    try:
        indices = list(map(int, indices_str.split(',')))
        if len(indices) == 5:
            return indices
    except ValueError:
        print("Error parsing reference indices.")
    return []


def process_frame_Tracking_Face(frame_path):
    try:
        return load_landmark_dlib(frame_path)
    except Exception as e:
        print(f"Error processing {frame_path}: {e}")
        return None

def main_process(source_video_path,driving_audio_path,mouth_region_size,custom_crop_radius,res_video_dir):
    # opt = LipSickInferenceOptions().parse_args()
    driving_audio_path = convert_audio_to_wav(driving_audio_path)

    # Ensure the res_video_dir is defined before using it
    res_video_dir = res_video_dir
    
    if not os.path.exists(source_video_path):
        raise Exception(f'Wrong video path: {source_video_path}')
    # if not os.path.exists(deepspeech_model_path):
    #     raise Exception('Please download the pretrained model of deepspeech')
    
    print('Extracting frames from video')
    video_frame_dir = source_video_path.replace('.mp4', '')
    if not os.path.exists(video_frame_dir):
        os.mkdir(video_frame_dir)
    video_size = extract_frames_from_video(source_video_path, video_frame_dir)

    ds_feature = DSModel.compute_audio_feature(driving_audio_path)
    res_frame_length = ds_feature.shape[0]
    ds_feature_padding = np.pad(ds_feature, ((2, 2), (0, 0)), mode='edge')

    print('Tracking Face')
    video_frame_path_list = glob.glob(os.path.join(video_frame_dir, '*.jpg'))
    video_frame_path_list.sort()
    num_workers = 5
    print(num_workers)
    with Pool(num_workers) as pool:
        video_landmark_data = pool.map(process_frame_Tracking_Face, video_frame_path_list)
    
    # Filter out any None results from the multiprocessing
    video_landmark_data_without_array = [data for data in video_landmark_data if data is not None]
    video_landmark_data = np.array(video_landmark_data_without_array)
    # video_landmark_data = np.array([load_landmark_dlib(frame) for frame in video_frame_path_list])

    print('Aligning frames with driving audio')
    video_frame_path_list_cycle = video_frame_path_list + video_frame_path_list[::-1]
    video_landmark_data_cycle = np.concatenate([video_landmark_data, np.flip(video_landmark_data, 0)], 0)
    video_frame_path_list_cycle_length = len(video_frame_path_list_cycle)
    if video_frame_path_list_cycle_length >= res_frame_length:
        res_video_frame_path_list = video_frame_path_list_cycle[:res_frame_length]
        res_video_landmark_data = video_landmark_data_cycle[:res_frame_length, :, :]
    else:
        divisor = res_frame_length // video_frame_path_list_cycle_length
        remainder = res_frame_length % video_frame_path_list_cycle_length
        res_video_frame_path_list = video_frame_path_list_cycle * divisor + video_frame_path_list_cycle[:remainder]
        res_video_landmark_data = np.concatenate([video_landmark_data_cycle] * divisor + [video_landmark_data_cycle[:remainder, :, :]], 0)
    res_video_frame_path_list_pad = [video_frame_path_list_cycle[0]] * 2 + res_video_frame_path_list + [video_frame_path_list_cycle[-1]] * 2
    res_video_landmark_data_pad = np.pad(res_video_landmark_data, ((2, 2), (0, 0), (0, 0)), mode='edge')
    assert ds_feature_padding.shape[0] == len(res_video_frame_path_list_pad) == res_video_landmark_data_pad.shape[0]
    pad_length = ds_feature_padding.shape[0]

    print('Selecting reference images based on input or randomly if unspecified')
    ref_img_list = []
    resize_w = int(mouth_region_size + mouth_region_size // 4)
    resize_h = int((mouth_region_size // 2) * 3 + mouth_region_size // 8)
    ref_index_list = random.sample(range(5, len(res_video_frame_path_list_pad) - 2), 5)

    print(f"Using reference frames at indices: {ref_index_list}")
    print('If each value has +5 added do not be alarmed it will -5 later')
    for ref_index in ref_index_list:
        if custom_crop_radius and custom_crop_radius > 0:
            crop_radius, crop_flag = custom_crop_radius, True
        else:
            crop_flag, crop_radius = compute_crop_radius(video_size, res_video_landmark_data_pad[ref_index - 5:ref_index, :, :])

        crop_radius_1_4 = crop_radius // 4
        ref_img = cv2.imread(res_video_frame_path_list_pad[ref_index - 3])[:, :, ::-1]
        ref_landmark = res_video_landmark_data_pad[ref_index - 3, :, :]
        ref_img_crop = ref_img[
                       ref_landmark[29, 1] - crop_radius:ref_landmark[29, 1] + crop_radius * 2 + crop_radius // 4,
                       ref_landmark[33, 0] - crop_radius - crop_radius // 4:ref_landmark[33, 0] + crop_radius + crop_radius // 4,
                       ]
        ref_img_crop = cv2.resize(ref_img_crop, (resize_w, resize_h))
        ref_img_crop = ref_img_crop / 255.0
        ref_img_list.append(ref_img_crop)
    ref_video_frame = np.concatenate(ref_img_list, axis=2)
    ref_img_tensor = torch.from_numpy(ref_video_frame).permute(2, 0, 1).unsqueeze(0).float().to('cuda')


    res_video_name = os.path.basename(source_video_path)[:-4] + '_facial_dubbing.mp4'
    res_video_path = os.path.join(res_video_dir, res_video_name)
    res_video_path = get_versioned_filename(res_video_path)  # Ensure unique filename

    videowriter = cv2.VideoWriter(res_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, video_size)

    if auto_mask:
        samelength_video_name = 'samelength.mp4'
        samelength_video_path = os.path.join(res_video_dir, samelength_video_name)
        samelength_video_path = get_versioned_filename(samelength_video_path)  # Ensure unique filename
        videowriter_samelength = cv2.VideoWriter(samelength_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, video_size)

    res_face_name = os.path.basename(source_video_path)[:-4] + '_facial_dubbing_face.mp4'
    res_face_path = os.path.join(res_video_dir, res_face_name)
    res_face_path = get_versioned_filename(res_face_path)  # Ensure unique filename

    videowriter_face = cv2.VideoWriter(res_face_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (resize_w, resize_h))

    for clip_end_index in range(5, pad_length, 1):
        sys.stdout.write(f'\rSynthesizing {clip_end_index - 5}/{pad_length - 5} frame')
        sys.stdout.flush()  # Make sure to flush the output buffer
        if not crop_flag:
            crop_radius = compute_crop_radius(video_size, res_video_landmark_data_pad[clip_end_index - 5:clip_end_index, :, :], random_scale=1.10)

        crop_radius_1_4 = crop_radius // 4
        frame_data = cv2.imread(res_video_frame_path_list_pad[clip_end_index - 3])[:, :, ::-1]
        frame_data_samelength = frame_data.copy()
        if auto_mask:
            videowriter_samelength.write(frame_data_samelength[:, :, ::-1])
        frame_landmark = res_video_landmark_data_pad[clip_end_index - 3, :, :]
        crop_frame_data = frame_data[
                          frame_landmark[29, 1] - crop_radius:frame_landmark[29, 1] + crop_radius * 2 + crop_radius_1_4,
                          frame_landmark[33, 0] - crop_radius - crop_radius_1_4:frame_landmark[33, 0] + crop_radius + crop_radius_1_4,
                          ]
        crop_frame_h, crop_frame_w = crop_frame_data.shape[0], crop_frame_data.shape[1]
        crop_frame_data = cv2.resize(crop_frame_data, (resize_w, resize_h)) / 255.0
        crop_frame_data[mouth_region_size // 2:mouth_region_size // 2 + mouth_region_size,
        mouth_region_size // 8:mouth_region_size // 8 + mouth_region_size, :] = 0

        crop_frame_tensor = torch.from_numpy(crop_frame_data).float().to('cuda').permute(2, 0, 1).unsqueeze(0)
        deepspeech_tensor = torch.from_numpy(ds_feature_padding[clip_end_index - 5:clip_end_index, :]).permute(1, 0).unsqueeze(0).float().to('cuda')

        with torch.no_grad():
            pre_frame = model(crop_frame_tensor, ref_img_tensor, deepspeech_tensor)
            pre_frame = pre_frame.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255
        videowriter_face.write(pre_frame[:, :, ::-1].copy().astype(np.uint8))
        pre_frame_resize = cv2.resize(pre_frame, (crop_frame_w, crop_frame_h))
        frame_data[
        frame_landmark[29, 1] - crop_radius:
        frame_landmark[29, 1] + crop_radius * 2,
        frame_landmark[33, 0] - crop_radius - crop_radius_1_4:
        frame_landmark[33, 0] + crop_radius + crop_radius_1_4,
        :] = pre_frame_resize[:crop_radius * 3, :, :]
        videowriter.write(frame_data[:, :, ::-1])
    videowriter.release()
    if auto_mask:
        videowriter_samelength.release()
    videowriter_face.release()

    if auto_mask:
        video_add_audio_path = os.path.join(res_video_dir, 'pre_blend.mp4')
    else:
        video_add_audio_path = os.path.join(res_video_dir, os.path.basename(source_video_path)[:-4] + '_LIPSICK.mp4')

    video_add_audio_path = get_versioned_filename(video_add_audio_path)  # Ensure unique filename

    cmd = f'ffmpeg -r 25 -i "{res_video_path}" -i "{driving_audio_path}" -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 "{video_add_audio_path}"'
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # Suppress FFmpeg logs
    # os.remove(res_video_path)  # Clean up intermediate files
    # os.remove(res_face_path)  # Clean up intermediate files

    if auto_mask:
        print('Auto Mask stage')
        samelength_video_path = os.path.join(res_video_dir, 'samelength.mp4')
        pre_blend_video_path = os.path.join(res_video_dir, 'pre_blend.mp4')

        # Call blend.py for blending and masking
        # cmd = [
        #     'python', 'utils/blend.py',
        #     '--samelength_video_path', samelength_video_path,
        #     '--pre_blend_video_path', pre_blend_video_path
        # ]
        # subprocess.call(cmd, shell=True)
        Processmain(samelength_video_path,pre_blend_video_path)




# @app.post("/process/")
# async def process_video(video: UploadFile = File(...), audio: UploadFile = File(...)):
#     video_path = f"temp_{video.filename}"
#     audio_path = f"temp_{audio.filename}"
#     output_path = "output.mp4"

#     # Save the uploaded files
#     with open(video_path, "wb") as buffer:
#         shutil.copyfileobj(video.file, buffer)

#     with open(audio_path, "wb") as buffer:
#         shutil.copyfileobj(audio.file, buffer)

#     output = main_process(video_path,audio_path,256,0,output_path)


#     # Clean up temporary files
#     # os.remove(video_path)
#     # os.remove(audio_path)

#     return FileResponse(output_path, media_type='video/mp4', filename="processed_video.mp4")
video_path = "Nishant_org.mp4"
audio_path = "speech.wav"
output_path = "inference_result"
main_process(video_path,audio_path,256,0,output_path)