import os

import numpy as np
import torch
from loguru import logger
from natsort import natsorted
from PIL import Image
from torch.autograd import Variable

from models.resnet import i3_res50


def load_frame(frame_file):
    data = Image.open(frame_file)
    data = np.array(data)
    data = data.astype(float)
    data = (data * 2 / 255) - 1
    assert(data.max() <= 1.0 and data.min() >= -1.0)
    return data


def load_rgb_batch(frames_dir, rgb_files, frame_indices):
    batch_data = np.zeros(frame_indices.shape + (224, 224, 3))
    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):
            batch_data[i, j, :, :, :] = load_frame(
                os.path.join(frames_dir, rgb_files[frame_indices[i][j]]))
    return batch_data


def forward_batch(i3d, b_data):
    b_data = b_data.transpose([0, 4, 1, 2, 3])
    b_data = torch.from_numpy(b_data)  # [b,c,t,h,w]
    with torch.no_grad():
        b_data = Variable(b_data.cuda()).float()
        inp = {'frames': b_data}
        features = i3d(inp)
    return features.cpu().numpy()


def extract_r3d_features(i3d, image_path):
    chunk_size = 32
    batch_size = 16
    clip_name = f'{image_path.split("/")[-3]}_{image_path.split("/")[-2]}'

    images = natsorted([i for i in os.listdir(image_path)])
    frame_cnt = len(images)
    assert(frame_cnt >= chunk_size)

    clipped_length = ((frame_cnt - chunk_size) // chunk_size) * chunk_size
    frame_indices = []
    for i in range(clipped_length // chunk_size + 1):
        frame_indices.append([j for j in range(i * chunk_size, i * chunk_size + chunk_size)])
    frame_indices = np.array(frame_indices)
    chunk_num = frame_indices.shape[0]
    batch_num = int(np.ceil(chunk_num / batch_size))
    frame_indices = np.array_split(frame_indices, batch_num, axis=0)

    full_features = []
    for batch_id in range(batch_num):
        batch_data = load_rgb_batch(image_path, images, frame_indices[batch_id])
        temp = forward_batch(i3d, batch_data)
        full_features.append(temp)

    # [batch_num, batch_size, ...] --> [batch_num x batch_size, ...]
    full_features = np.concatenate(full_features, axis=0)
    
    for i, feature in enumerate(full_features):
        np.save(f'{r3d_feature_path}{clip_name}_clip{i+1}', feature)
    logger.info(f'{clip_name}, {frame_cnt:4d} frames, {chunk_num} clips')


if __name__ == "__main__":
    logger.add(f'logs/extract_r3d_clip.txt', format='{message}', level='INFO')

    pretrained_path = 'pretrained/i3d_r50_kinetics.pth'
    hhi_kinect_cropped_path = 'data/hhi_kinect_session_cropped/'
    r3d_feature_path = 'features/r3d_features_clip/'
    os.makedirs(r3d_feature_path, exist_ok=True)

    i3d = i3_res50(400, pretrained_path)
    i3d.cuda()
    i3d.train(False)

    sessions = natsorted(os.listdir(hhi_kinect_cropped_path))
    p_num = 2
    for session in sessions:
        for p_idx in range(p_num):
            image_path = f'{hhi_kinect_cropped_path}{session}/{p_idx+1}/'
            extract_r3d_features(i3d, image_path)
