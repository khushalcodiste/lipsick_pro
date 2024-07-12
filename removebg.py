import typing as typ
from PIL import Image
from tqdm import tqdm
import cv2
from rembg import remove, new_session
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from gfpgan import GFPGANer

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


MODEL_CHECKSUM_DISABLED=True

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
    with Pool(cpu_count()) as p:
        list(tqdm(p.imap(restore_face, old_frame), total=len(old_frame), desc="Processing frames for enhancement"))
    return old_frame


def process_frame(input_path):
    
    input = Image.open(input_path)
    session = new_session('u2net')
    output = remove(input, session=session)
    os.remove(input_path)
    input_path = input_path.split(".")
    output.save(input_path[0] + '.png')

def RemoveFramemain(video_path, save_dir):
    samples = extract_frames_from_video(video_path, save_dir)
    with Pool(1) as p:
        list(tqdm(p.imap(process_frame, samples), total=len(samples), desc="Processing frames for background removal"))


    # with Pool(2) as p:
    #     list(tqdm(p.imap(process_frame, samples), total=len(samples)))



# if __name__ == "__main__":
#     RemoveFramemain()
