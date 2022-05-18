import os

import numpy as np
import cv2
import pandas as pd
from PIL import Image
from natsort import natsorted
from torch import clip_
from torch.utils.data import Dataset


class MHHRIDataet(Dataset):
    """MHHRI dataset, used for training from scratch."""
    pass


def create_labels(files, labels, trait):
    labels = labels[['session', trait]]
    label_out = {}

    for file in files:
        file_name = file.split('.')[0]
        sess_name = f"{file_name.split('_')[0]}_{file_name.split('_')[1]}_{file_name.split('_')[2]}"
        label_out[file] = labels[labels['session'] == sess_name][trait].values[0]

    return label_out


def create_mhhri(self_body_train_files, self_body_test_files,
                 self_face_train_files, self_face_test_files,
                 interact_body_train_files, interact_body_test_files,
                 interact_face_train_files, interact_face_test_files,
                 task, label_type, trait):

    # select ground truth file: [class, reg], [acq, self]
    label_path = f'data/annotations/{task}_{label_type}_session_norm.csv'
    labels = pd.read_csv(label_path)

    train_label = create_labels(self_body_train_files, labels, trait)
    test_label = create_labels(self_body_test_files, labels, trait)
    # with open(f'{fold}_{trait}.csv', 'w') as f:
    #     for key in test_label.keys():
    #         f.write("%s, %s\n" % (key, test_label[key]))

    train_data = MHHRI(self_body_train_files, self_face_train_files,
                       interact_body_train_files, interact_face_train_files,
                       train_label)
    test_data = MHHRI(self_body_test_files, self_face_test_files,
                      interact_body_test_files, interact_face_test_files,
                      test_label)

    return train_data, test_data


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


def data_sampler(body_path, face_path):
    # set parameters
    chunk_size_body = 32
    chunk_size_face = 4

    # sample body images --> 32 frames
    sessions = natsorted(os.listdir(body_path))
    persons = ['1', '2']
    for session in sessions:
        for person in persons:
            image_folder = body_path + session + '/' + person + '/'
            images = natsorted([i for i in os.listdir(image_folder)])
            frame_num = len(images)
            clip_num = frame_num // chunk_size_body
            frame_num = clip_num * chunk_size_body
            images = images[:frame_num]
            print(np.shape(images), clip_num)

            for clip_idx in range(0, frame_num, chunk_size_body):
                image_sequence = images[clip_idx:clip_idx+chunk_size_body]
                print(image_sequence)
                clip_data = np.zeros((chunk_size_body, 224, 224, 3))
                for frame_idx in range(chunk_size_body):
                    clip_data[frame_idx, :, :, :] = cv2.imread(
                        image_folder + image_sequence[frame_idx]
                    )
                print(clip_data)
                break
            break
        break

    np.save('clip_data', clip_data)
        #     for i in range(clipped_length // chunk_size_body + 1):
        #         frame_indices.append([j for j in range(i * chunk_size_body, i * chunk_size_body + chunk_size_body)])
        #     frame_indices = np.array(frame_indices)
        #     chunk_num = frame_indices.shape[0]
        #     batch_num = int(np.ceil(chunk_num / batch_size))
        #     frame_indices = np.array_split(frame_indices, batch_num, axis=0)
        #     print(np.shape(frame_indices))

        #     full_features = []
        #     for batch_id in range(batch_num):
        #         batch_data = load_rgb_batch(image_path, images, frame_indices[batch_id])
        #         temp = forward_batch(i3d, batch_data)
        #         full_features.append(temp)


            # # [batch_num, batch_size, ...] --> [batch_num x batch_size, ...]
            # full_features = np.concatenate(full_features, axis=0)
            
            # for i, feature in enumerate(full_features):
            #     np.save(f'{r3d_feature_path}{clip_name}_clip{i+1}', feature)
            # logger.info(f'{clip_name}, {frame_num:4d} frames, {chunk_num} clips')


    # sample face images --> 2 frames
    pass


if __name__ == '__main__':
    # b = np.load('features/r3d_features/S01_kinect_1_clip1.npy')
    # f = np.load('features/face_features_fixed/S01_ego_1_clip1.npy')
    # print(b.shape, f.shape)

    # fold_num = 6
    # self_body_data_list = np.load('data/data_list/acq_self_body.npy', allow_pickle=True)
    # self_face_data_list = np.load('data/data_list/acq_self_face.npy', allow_pickle=True)
    # interact_body_data_list = np.load('data/data_list/acq_interact_body.npy', allow_pickle=True)
    # interact_face_data_list = np.load('data/data_list/acq_interact_face.npy', allow_pickle=True)

    # print(self_body_data_list[0][:10])
    # print(self_face_data_list[0][:10])
    # print(interact_body_data_list[0][:10])
    # print(interact_face_data_list[0][:10])
    # print(np.shape(self_body_data_list), np.shape(self_face_data_list), np.shape(interact_body_data_list), np.shape(interact_face_data_list))

    # task = 'acq'
    # label_type = 'reg'
    # traits = ['O', 'C', 'E', 'A', 'N']

    # for fold in range(fold_num):
    #     self_body_test_files = self_body_data_list[fold]
    #     self_body_train_list = np.delete(self_body_data_list, fold, axis=0)
    #     self_body_train_files = [item for row in self_body_train_list for item in row]
    #     # print(f'fold {fold}: train_num={len(self_body_train_files)}, test_num={len(self_body_test_files)}')
    #     print()
    #     print(self_body_train_files[:10])
    #     print(self_body_train_files[-10:])
    #     print(self_body_test_files[:10])
    #     print(self_body_test_files[-10:])
    #     print(np.shape(self_body_train_files), np.shape(self_body_test_files))

    #     self_face_test_files = self_face_data_list[fold]
    #     self_face_train_list = np.delete(self_face_data_list, fold, axis=0)
    #     self_face_train_files = [item for row in self_face_train_list for item in row]
    #     # print(f'fold {fold}: train_num={len(self_face_train_files)}, test_num={len(self_face_test_files)}')

    #     interact_body_test_files = interact_body_data_list[fold]
    #     interact_body_train_list = np.delete(interact_body_data_list, fold, axis=0)
    #     interact_body_train_files = [item for row in interact_body_train_list for item in row]
    #     # print(f'fold {fold}: train_num={len(interact_body_train_files)}, test_num={len(interact_body_test_files)}')

    #     interact_face_test_files = interact_face_data_list[fold]
    #     interact_face_train_list = np.delete(interact_face_data_list, fold, axis=0)
    #     interact_face_train_files = [item for row in interact_face_train_list for item in row]
    #     # print(f'fold {fold}: train_num={len(interact_face_train_files)}, test_num={len(interact_face_test_files)}')

    #     for trait in traits:
    #         train_data, test_data = create_mhhri(
    #             self_body_train_files, self_body_test_files,
    #             self_face_train_files, self_face_test_files,
    #             interact_body_train_files,  interact_body_test_files,
    #             interact_face_train_files, interact_face_test_files,
    #             task=task, label_type=label_type, trait=trait
    #         )
    #         break
    #     break

    # body_path = 'data/hhi_kinect_session_cropped/'
    # face_path = 'data/hhi_ego_face_fixed/'
    # data_sampler(body_path, face_path)

    clip_data = np.load('clip_data.npy')
    print(np.shape(clip_data))
    print(clip_data)
    for i in range(len(clip_data)):
        cv2.imwrite(f'img_{i}.png', clip_data[i])

    pass
