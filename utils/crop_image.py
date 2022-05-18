import os

import cv2
import imutils
import numpy as np
import pandas as pd
from natsort import natsorted


def write_image(image_name, person_1, person_2, out_path_1, out_path_2):
    box_1_h, box_1_w = person_1.shape[:-1]
    box_2_h, box_2_w = person_2.shape[:-1]
    new_image_1 = np.zeros((224, 224, 3))
    new_image_2 = np.zeros((224, 224, 3))

    if box_1_h >= box_1_w:
        temp_image_1 = imutils.resize(person_1, height=224)
        box_1_h, box_1_w = temp_image_1.shape[:-1]
        new_image_1[:, int((224-box_1_w)/2):int(224 - (224-box_1_w)/2), :] = temp_image_1
    if box_1_h < box_1_w:
        temp_image_1 = imutils.resize(person_1, width=224)
        box_1_h, box_1_w = temp_image_1.shape[:-1]
        new_image_1[int((224-box_1_h)/2):int(224 - (224-box_1_h)/2), :, :] = temp_image_1
    if box_2_h >= box_2_w:
        temp_image_2 = imutils.resize(person_2, height=224)
        box_2_h, box_2_w = temp_image_2.shape[:-1]
        new_image_2[:, int((224-box_2_w)/2):int(224 - (224-box_2_w)/2), :] = temp_image_2
    if box_2_h < box_2_w:
        temp_image_2 = imutils.resize(person_2, width=224)
        box_2_h, box_2_w = temp_image_2.shape[:-1]
        new_image_2[int((224-box_2_h)/2):int(224 - (224-box_2_h)/2), :, :] = temp_image_2

    box_1_h, box_1_w = new_image_1.shape[:-1]
    box_2_h, box_2_w = new_image_2.shape[:-1]
    # print(box_1_h, box_1_w, box_2_h, box_2_w)

    cv2.imwrite(f'{out_path_1}{image_name}', new_image_1)
    cv2.imwrite(f'{out_path_2}{image_name}', new_image_2)


def crop_image():
    hhi_kinect_path = 'data/hhi_kinect_session/'
    bounding_box_path = 'features/yolox/'
    hhi_kinect_sessions = natsorted(os.listdir(hhi_kinect_path))
    print(hhi_kinect_sessions)

    for hhi_kinect_session in hhi_kinect_sessions:
        print(f'processing {hhi_kinect_session}')
        bounding_box_file = bounding_box_path + hhi_kinect_session + '.txt'
        boxes = pd.read_csv(bounding_box_file, header=None)
        boxes.columns = ['frame', 'id', 'x1', 'y1', 'x2', 'y2']

        boxes_1 = boxes[boxes['id']==1]
        boxes_1_min_x1, boxes_1_min_y1 = boxes_1.min()[2], boxes_1.min()[3]
        boxes_1_max_x2, boxes_1_max_y2 = boxes_1.max()[4], boxes_1.max()[5]
        box_1 = [boxes_1_min_x1, boxes_1_min_y1, boxes_1_max_x2, boxes_1_max_y2]
        box_1[0] = max(box_1[0], 0)
        box_1[1] = max(box_1[1], 0)
        box_1[2] = min(box_1[2], 640)
        box_1[3] = max(box_1[3], 480)

        boxes_2 = boxes[boxes['id']==2]
        boxes_2_min_x1, boxes_2_min_y1 = boxes_2.min()[2], boxes_2.min()[3]
        boxes_2_max_x2, boxes_2_max_y2 = boxes_2.max()[4], boxes_2.max()[5]
        box_2 = [boxes_2_min_x1, boxes_2_min_y1, boxes_2_max_x2, boxes_2_max_y2]
        box_2[0] = max(box_2[0], 0)
        box_2[1] = max(box_2[1], 0)
        box_2[2] = min(box_2[2], 640)
        box_2[3] = max(box_2[3], 480)

        image_path = f'data/hhi_kinect_session/{hhi_kinect_session}/'
        cropped_image_path_1 = f'data/hhi_kinect_session_cropped/{hhi_kinect_session}/1/'
        cropped_image_path_2 = f'data/hhi_kinect_session_cropped/{hhi_kinect_session}/2/'
        os.makedirs(cropped_image_path_1, exist_ok=True)
        os.makedirs(cropped_image_path_2, exist_ok=True)
        images = natsorted(os.listdir(image_path))

        for image in images:
            image_name = image
            image = cv2.imread(image_path + image)
            person_1 = image[box_1[1]:box_1[3], box_1[0]:box_1[2], :]
            person_2 = image[box_2[1]:box_2[3], box_2[0]:box_2[2], :]
            # print(np.shape(image), np.shape(person_1), np.shape(person_2))
            write_image(image_name, person_1, person_2, cropped_image_path_1, cropped_image_path_2)


def process_bounding_box():
    bounding_box_path = 'features/yolox/'
    bounding_box_files = natsorted(os.listdir(bounding_box_path))
    for bounding_box_file in bounding_box_files:
        boxes = pd.read_csv(bounding_box_path+bounding_box_file, header=None)
        boxes.columns = ['frame', 'id', 'x1', 'y1', 'x2', 'y2']
        id_freq = boxes['id'].value_counts()
        p_num = len(id_freq)
        print(f'{bounding_box_file.split(".")[0]} has {p_num} targets')
        print(id_freq)

    # change id
    boxes = pd.read_csv(bounding_box_path + 'S12_kinect.txt', header=None)
    boxes.columns = ['frame', 'id', 'x1', 'y1', 'x2', 'y2']
    boxes.loc[boxes['id'] == 3, 'id'] = 2
    boxes.loc[boxes['id'] == 5, 'id'] = 2
    boxes.loc[boxes['id'] == 8, 'id'] = 2
    id_freq = boxes['id'].value_counts()
    p_num = len(id_freq)
    print(f'{bounding_box_file.split(".")[0]} has {p_num} targets')
    print(id_freq)

    # change left and right 
    bounding_box_files = ['S01_kinect.txt', 'S04_kinect.txt', 'S05_kinect.txt', 'S09_kinect.txt', 'S11_kinect.txt']
    for bounding_box_file in bounding_box_files:
        boxes = pd.read_csv(bounding_box_path+bounding_box_file, header=None)
        boxes.columns = ['frame', 'id', 'x1', 'y1', 'x2', 'y2']
        boxes.loc[boxes['id'] == 1, 'id'] = 3
        boxes.loc[boxes['id'] == 2, 'id'] = 1
        boxes.loc[boxes['id'] == 3, 'id'] = 2
        print(boxes)
        boxes.to_csv(bounding_box_file, index=False)


if __name__ == "__main__":
    # process_bounding_box()
    crop_image()
