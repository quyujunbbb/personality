import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from natsort import natsorted

from models import nets_scratch


def read_images(folder, files):
    chunk_size = len(files)
    imgs = np.zeros((1, chunk_size, 224, 224, 3))
    for idx, file in enumerate(files):
        img = cv2.imread(folder + file)
        imgs[:, idx, :, :, :] = img
    print(np.shape(imgs))

    return imgs


def get_feature_map():
    body_1_files = natsorted(os.listdir('visualize/sample_body/1/'))
    body_2_files = natsorted(os.listdir('visualize/sample_body/2/'))
    face_1_files = natsorted(os.listdir('visualize/sample_face/1/'))
    face_2_files = natsorted(os.listdir('visualize/sample_face/2/'))

    body_1 = read_images('visualize/sample_body/1/', body_1_files)
    body_2 = read_images('visualize/sample_body/2/', body_2_files)
    face_1 = read_images('visualize/sample_face/1/', face_1_files)
    face_2 = read_images('visualize/sample_face/2/', face_2_files)

    body_1 = torch.tensor(body_1).permute(0, 4, 1, 2, 3)
    body_2 = torch.tensor(body_2).permute(0, 4, 1, 2, 3)
    face_1 = torch.tensor(face_1).permute(0, 1, 4, 2, 3).reshape(-1, 3, 224, 224)
    face_2 = torch.tensor(face_2).permute(0, 1, 4, 2, 3).reshape(-1, 3, 224, 224)

    print(body_1.size(), face_1.size(), body_2.size(), face_2.size())

    body_1 = body_1.to(torch.float).cuda()
    body_2 = body_2.to(torch.float).cuda()
    face_1 = face_1.to(torch.float).cuda()
    face_2 = face_2.to(torch.float).cuda()

    r3d = nets_scratch.ResNet3D(num_classes=400)
    r3d.load_state_dict(torch.load('pretrained/i3d_r50_kinetics.pth'))
    r3d.cuda()

    dmue = nets_scratch.DMUE(bnneck=True)
    dmue.load_param()
    dmue = dmue.cuda()

    net = nets_scratch.MyModel(r3d, dmue)
    net.cuda()
    out = net(body_1, face_1, body_2, face_2)
    print(out.size())


def plot_map():
    path = 'visualize/feature_maps/npy/body/'
    files = natsorted(os.listdir(path))

    for file in files:
        map_name = file.split('.')[0]
        feature_map = torch.tensor(np.load(path + file))

        feature_map = feature_map.squeeze(0)
        print(feature_map.shape)
        gray_scale = torch.sum(feature_map,0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed = gray_scale.data.cpu().numpy()

        fig = plt.figure(figsize=(30, 50))
        for i in range(len(processed)):
            a = fig.add_subplot(5, 4, i+1)
            imgplot = plt.imshow(processed[i])
        plt.savefig(f'visualize/feature_maps/plot/{map_name}.png', bbox_inches='tight')

    path = 'visualize/feature_maps/npy/face/'
    files = natsorted(os.listdir(path))
    for file in files:
        map_name = file.split('.')[0]
        feature_map = torch.tensor(np.load(path + file))

        feature_map = feature_map.squeeze(0).permute(1, 0, 2, 3)
        print(feature_map.shape)
        gray_scale = torch.sum(feature_map,0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed = gray_scale.data.cpu().numpy()

        fig = plt.figure(figsize=(30, 50))
        for i in range(len(processed)):
            a = fig.add_subplot(5, 4, i+1)
            imgplot = plt.imshow(processed[i])
        plt.savefig(f'visualize/feature_maps/plot/{map_name}.png', bbox_inches='tight')


if __name__ == "__main__":
    # get_feature_map()
    plot_map()
