# """
# All models are here
# u2net (download, source): A pre-trained model for general use cases.
# u2netp (download, source): A lightweight version of u2net model.
# u2net_human_seg (download, source): A pre-trained model for human segmentation.
# silueta (download, source): Same as u2net but the size is reduced to 43Mb.
# u2net_cloth_seg (download, source): A pre-trained model for Cloths Parsing from human portrait. Here clothes are parsed into 3 category: Upper body, Lower body and Full body.
# isnet-general-use (download, source): A new pre-trained model for general use cases.
# isnet-anime (download, source): A high-accuracy segmentation for anime character.
# sam (download encoder, download decoder, source): A pre-trained model for any use cases.
# """
# import typing as typ

# from PIL import Image
# from tqdm import tqdm
# import cv2
# from rembg import remove, new_session
# import os
# # MODELS: typ.List[str] = [
# #     "u2net",
# #     "u2netp",
# #     "u2net_human_seg",
# #     "silueta",
# #     "u2net_cloth_seg",
# #     "isnet-general-use",
# #     "isnet-anime",
# #     "sam",
# # ]

# def extract_frames_from_video(video_path, save_dir):
#     video_capture = cv2.VideoCapture(video_path)
#     frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
#     old_frame = []
#     for i in range(frames):
#         ret, frame = video_capture.read()
#         if not ret:
#             break
#         result_path = os.path.join(save_dir, str(i).zfill(6) + '.jpg')
#         old_frame.append(result_path)
#         cv2.imwrite(result_path, frame)
#     return old_frame


# def main() -> None:
#     """Run main function."""
#     video_path = "inference_result\\LipSick_Blend_old.mp4"
#     save_dir = "inference_result\\tempunet"
#     # samples = [
#     #      'Nishant_org\\000001.jpg'
#     # ]
#     # for model_name in tqdm(MODELS):
#     samples = extract_frames_from_video(video_path,save_dir)
#     for input_path in samples:
#         input = Image.open(input_path)
#         session = new_session('u2net')
#         output = remove(input, session=session)
#         os.remove(input_path)
#         input_path = input_path.split(".")
#         output.save(input_path[0]+'.png')


# if __name__ == "__main__":
#     main()

"""
All models are here
u2net (download, source): A pre-trained model for general use cases.
u2netp (download, source): A lightweight version of u2net model.
u2net_human_seg (download, source): A pre-trained model for human segmentation.
silueta (download, source): Same as u2net but the size is reduced to 43Mb.
u2net_cloth_seg (download, source): A pre-trained model for Cloths Parsing from human portrait. Here clothes are parsed into 3 category: Upper body, Lower body and Full body.
isnet-general-use (download, source): A new pre-trained model for general use cases.
isnet-anime (download, source): A high-accuracy segmentation for anime character.
sam (download encoder, download decoder, source): A pre-trained model for any use cases.
"""

import typing as typ
from PIL import Image
from tqdm import tqdm
import cv2
from rembg import remove, new_session
import os
from multiprocessing import Pool, cpu_count

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
    return old_frame

def process_frame(input_path):
    input = Image.open(input_path)
    session = new_session('u2net')
    output = remove(input, session=session)
    os.remove(input_path)
    input_path = input_path.split(".")
    output.save(input_path[0] + '.png')

def main() -> None:
    """Run main function."""
    video_path = "inference_result\\LipSick_Blend_old.mp4"
    save_dir = "inference_result\\tempunet"
    samples = extract_frames_from_video(video_path, save_dir)
    
    with Pool(cpu_count()) as p:
        list(tqdm(p.imap(process_frame, samples), total=len(samples)))

if __name__ == "__main__":
    main()
