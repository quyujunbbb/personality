import os

import cv2
import numpy as np
from natsort import natsorted


def move_figs():
    """Move sample figures from face and body images.
    """
    body_path = 'data/hhi_kinect_session_cropped/'
    for session in natsorted(os.listdir(body_path)):
        for user in natsorted(os.listdir(body_path+session+'/')):
            imgs = natsorted(os.listdir(body_path+session+'/'+user+'/'))
            img = cv2.imread(body_path+session+'/'+user+'/'+imgs[0])
            cv2.imwrite(f'body_{session}_{user}.png', img)

    face_path = 'data/hhi_ego_face_fixed/'
    for session in natsorted(os.listdir(face_path)):
        for user in natsorted(os.listdir(face_path+session+'/')):
            imgs = natsorted(os.listdir(face_path+session+'/'+user+'/'))
            img = cv2.imread(face_path+session+'/'+user+'/'+imgs[0])
            cv2.imwrite(f'face_{session}_{user}.png', img)


def image2numpy(input_path, output_path):
    images_out = []
    images = natsorted([i for i in os.listdir(input_path)])
    totle_image_number = len(images)
    images = images[0:totle_image_number:5]
    sampled_image_number = len(images)
    print(f'total {totle_image_number} images, samples {sampled_image_number} images')
    for image in images:
        temp = cv2.imread(input_path + image)
        temp = temp[:, 420:1500, :]
        images_out.append(temp)
    images_out = np.array(images_out)
    print(images_out.shape)

    np.save(output_path, images_out)


def compare_image_crops():
    image_path = 'data/hhi_kinect/S01_kinect_clip01/kinect2_color_14_05_17_444.png'
    image = cv2.imread(image_path)
    print(np.shape(image))
    image_center_crop = image[:, 80:560, :]
    image_left_crop = image[:, :480, :]
    image_right_crop = image[:, 160:, :]
    cv2.imwrite(f'image_center_crop.jpg', image_center_crop)
    cv2.imwrite(f'image_left_crop.jpg', image_left_crop)
    cv2.imwrite(f'image_right_crop.jpg', image_right_crop)


def video2image(target_session):
    print(f'extract frames from videos')
    ego_video_path = 'data/hhi_ego/'
    ego_image_out_path = 'data/hhi_ego_images/'
    videos = natsorted(os.listdir(ego_video_path))
    target_session = f'S{target_session:02d}'
    print(target_session)

    for video in videos:
        video_name = video.split('.')[0].split('_')
        session = video_name[0]
        if session != target_session: break
        clip = video_name[-1]
        user_2 = video_name[4]
        image_out_folder = f'{session}_ego'
        image_path = ego_image_out_path + image_out_folder + '/' + user_2 + '/'
        os.makedirs(image_path, exist_ok=True)

        print(f'processing {video}, extracting to {image_path}')
        vidcap = cv2.VideoCapture(ego_video_path+video)
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(f'{image_path}{clip}_{count}.jpg', image)
            success, image = vidcap.read()
            count += 1


def sample_image(session):
    print(f'sample frames from all images')
    # frames_in_session = [19487,19384,25044,5797,4017,3195,9860,4790,14303,14830,13846,10746]
    frames_in_session = [2432,2420,3128,724,500,396,1232,596,1784,1852,1728,1340]
    frame_num = frames_in_session[session-1]

    session_folder = f'data/hhi_ego_images/S{session:02d}_ego/'
    user_folders = natsorted(os.listdir(session_folder))
    for user_folder in user_folders:
        in_folder = session_folder + user_folder + '/'
        out_folder = f'data/hhi_ego_images_sampled/S{session:02d}_ego/{user_folder}/'
        os.makedirs(out_folder, exist_ok=True)

        image_files = natsorted(os.listdir(in_folder))
        image_num = len(image_files)
        print(f'{image_num} --> {frame_num}')

        # from [0, image_num=1] evenly sample frame_num images
        index_list = np.linspace(0, image_num-1, frame_num).astype(int).tolist()
        for index in index_list:
            image = cv2.imread(in_folder + image_files[index])
            cv2.imwrite(f'{out_folder}{image_files[index]}', image)
        

def mhhri_ego_video_to_sampled_image():
    # session_num = 12
    # for i in range(session_num):
    #     session = i+1
    #     print(f'processing S{session:02d} ...')
    #     video2image(session)
    #     sample_image(session)
    session = 11
    print(f'processing S{session:02d} ...')
    video2image(session)
    sample_image(session)


if __name__ == '__main__':
    mhhri_ego_video_to_sampled_image()
    pass
