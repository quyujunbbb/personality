import os
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted


def filename2time(image_name):
    name = image_name.split('.')[0].split('_')
    h, m, s, ms = name[-4], name[-3], name[-2], name[-1]
    return h, m, s, ms


def draw_frame_interval(clips):
    kinect_folder_path = 'data/hhi_kinect/'
    output_path = 'data/data_summary/frame_interval.png'

    fig, axs = plt.subplots(2, 2, layout='constrained')
    fig.suptitle("frame interval")
    for clip, ax in zip(clips, axs.ravel()):
        duration = []
        image_path = kinect_folder_path + clip + '/'
        images = natsorted(os.listdir(image_path))
        frame_count = len(images)
        for idx in range(frame_count-1):
            h_0, m_0, s_0, ms_0 = filename2time(images[idx])
            time_0 = datetime.strptime(f'{h_0}:{m_0}:{s_0}.{ms_0}', '%H:%M:%S.%f')
            h_1, m_1, s_1, ms_1 = filename2time(images[idx+1])
            time_1 = datetime.strptime(f'{h_1}:{m_1}:{s_1}.{ms_1}', '%H:%M:%S.%f')
            duration.append((time_1 - time_0).total_seconds())
        ax.scatter(range(frame_count-1), duration, s=1)
        ax.set_xlabel(f'{clip}')
        fig.savefig(output_path)


def draw_clip_info():
    ego_csv_path = 'data/data_summary/hhi_ego_concate.csv'
    kinect_csv_path = 'data/data_summary/hhi_kinect.csv'
    ego_info = pd.read_csv(ego_csv_path)
    kinect_info = pd.read_csv(kinect_csv_path)
    # =============================================================
    #       frame_1  length_1  frame_2  length_2  frame_k  length_k
    # -------------------------------------------------------------
    # mean  3536.43     59.00  3540.16     59.06  1009.02     59.38
    # std   2185.03     36.45  2185.09     36.45   754.32     36.41
    # min       417      6.96      401      6.69       52      7.24
    # max     15328    255.72    15297    255.21     3992    255.24
    # =============================================================
    x = range(144)
    frame_1, length_1 = ego_info['frame_1'], ego_info['length_1']
    frame_2, length_2 = ego_info['frame_2'], ego_info['length_2']
    frame_k, length_k = kinect_info['frame'], kinect_info['length']
    fps_1 = ego_info['frame_1'] / ego_info['length_1']
    fps_2 = ego_info['frame_2'] / ego_info['length_2']
    fps_k = kinect_info['frame'] / kinect_info['length']

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained')
    ax1.plot(x, frame_1, label="ego 1")
    ax1.plot(x, frame_2, label="ego 2")
    ax1.plot(x, frame_k, label="kinect")
    ax1.set_xlabel("clip")
    ax1.set_ylabel("frame no.")
    ax1.legend()

    ax2.plot(x, length_1, label="ego 1")
    ax2.plot(x, length_2, label="ego 2")
    ax2.plot(x, length_k, label="kinect")
    ax2.set_xlabel("clip")
    ax2.set_ylabel("length")
    ax2.legend()

    ax3.scatter(x, fps_1, label="ego 1", s=1)
    ax3.scatter(x, fps_2, label="ego 2", s=1)
    ax3.scatter(x, fps_k, label="kinect", s=1)
    ax3.set_xlabel("clip")
    ax3.set_ylabel("fps")
    ax3.legend()
    fig.savefig(f'data/data_summary/clip_info.png')


def get_ego_info():
    ego_video_path = 'data/hhi_ego/'
    output_path = 'data/data_summary/hhi_ego.csv'

    # func 1: get ego-view video info
    videos = natsorted(os.listdir(ego_video_path))
    ego_info = pd.DataFrame(columns=['clip', 'frame', 'length', 'fps'])
    for video in videos:
        vidcap = cv2.VideoCapture(ego_video_path + video)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        vidcap.release()

        row = {'clip':video.split(".")[0], 'frame':frame_count, 'length':duration, 'fps':fps}
        ego_info = ego_info.append(row, ignore_index=True)
    ego_info.to_csv(output_path, index=False)


def get_kinect_info():
    kinect_folder_path = 'data/hhi_kinect/'
    output_path = 'data/data_summary/hhi_kinect.csv'

    # func 1: get kinect info
    image_folders = natsorted(os.listdir(kinect_folder_path))
    kinect_info = pd.DataFrame(columns=['clip', 'frame', 'length', 'fps'])
    for image_folder in image_folders:
        image_path = kinect_folder_path + image_folder + '/'
        images = natsorted(os.listdir(image_path))
        frame_count = len(images)

        image_begin = images[0]
        h_0, m_0, s_0, ms_0 = filename2time(image_begin)
        time_0 = datetime.strptime(f'{h_0}:{m_0}:{s_0}.{ms_0}', '%H:%M:%S.%f')

        image_end = images[-1]
        h_1, m_1, s_1, ms_1 = filename2time(image_end)
        time_1 = datetime.strptime(f'{h_1}:{m_1}:{s_1}.{ms_1}', '%H:%M:%S.%f')

        duration = (time_1 - time_0).total_seconds()
        fps = frame_count / duration

        row = {'clip':image_folder, 'frame':frame_count, 'length':duration, 'fps':fps}
        kinect_info = kinect_info.append(row, ignore_index=True)
    kinect_info.to_csv(output_path, index=False)

    # func 2: check the duration of two adjacent clips in same session
    # for folder_index in range(len(image_folders)-1):
    #     session = image_folders[folder_index].split('_')[0]

    #     clip_1 = image_folders[folder_index].split('_')[-1]
    #     folder_1 = kinect_folder_path + image_folders[folder_index] + '/'
    #     images_1 = natsorted(os.listdir(folder_1))
    #     h_1, m_1, s_1, ms_1 = filename2time(images_1[-1])
    #     time_1 = datetime.strptime(f'{h_1}:{m_1}:{s_1}.{ms_1}', '%H:%M:%S.%f')

    #     clip_2 = image_folders[folder_index+1].split('_')[-1]
    #     folder_2 = kinect_folder_path + image_folders[folder_index+1] + '/'
    #     images_2 = natsorted(os.listdir(folder_2))
    #     h_2, m_2, s_2, ms_2 = filename2time(images_2[0])
    #     time_2 = datetime.strptime(f'{h_2}:{m_2}:{s_2}.{ms_2}', '%H:%M:%S.%f')

    #     print(f'{session} {clip_1}-{clip_2} '
    #           f'{h_1}:{m_1}:{s_1}:{ms_1}-{h_2}:{m_2}:{s_2}:{ms_2} '
    #           f'{(time_2 - time_1).total_seconds()}')


if __name__ == "__main__":
    # get_kinect_info()
    # get_ego_info()

    # draw_clip_info()

    sample_kinect_clips = ['S01_kinect_clip10', 'S04_kinect_clip05',
                           'S05_kinect_clip08', 'S10_kinect_clip11']
    draw_frame_interval(sample_kinect_clips)
