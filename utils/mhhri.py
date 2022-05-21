import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from natsort import natsorted
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MHHRIDataset(Dataset):
    """MHHRI dataset, used for training from scratch."""

    def __init__(self, s_body_train_files, s_face_train_files,
        i_body_train_files, i_face_train_files,
        labels, transform=None):
        self.s_body_train_files = s_body_train_files
        self.s_face_train_files = s_face_train_files
        self.i_body_train_files = i_body_train_files
        self.i_face_train_files = i_face_train_files
        self.labels = labels
        self.T = transform

    def __getitem__(self, index):
        s_body_file = self.s_body_train_files[index]
        s_face_file = self.s_face_train_files[index]
        i_body_file = self.i_body_train_files[index]
        i_face_file = self.i_face_train_files[index]

        s_body = np.load(f'data/hhi_kinect_body_np/{s_body_file}').astype('float32')
        s_face = np.load(f'data/hhi_ego_face_np/{s_face_file}').astype('float32')
        i_body = np.load(f'data/hhi_kinect_body_np/{i_body_file}').astype('float32')
        i_face = np.load(f'data/hhi_ego_face_np/{i_face_file}').astype('float32')
        label = self.labels[s_body_file]
        # print(np.shape(s_body), np.shape(s_face))

        if self.T is not None:
            s_body = self.transform_body(s_body)
            s_face = self.transform_face(s_face)
            i_body = self.transform_body(i_body)
            i_face = self.transform_face(i_face)

        # for i in range(32):
        #     cv2.imwrite(f's_body_{i}.png', s_body[i])
        #     cv2.imwrite(f'i_body_{i}.png', i_body[i])
        # for i in range(4):
        #     cv2.imwrite(f's_face_{i}.png', s_face[i])
        #     cv2.imwrite(f'i_face_{i}.png', i_face[i])

        return s_body, s_face, i_body, i_face, label

    def __len__(self):
        return len(self.s_body_train_files)

    def transform_body(self, imgs):
        imgs_T = self.T(
            image=imgs[0],  img1=imgs[1],   img2=imgs[2],   img3=imgs[3],
            img4=imgs[4],   img5=imgs[5],   img6=imgs[6],   img7=imgs[7],
            img8=imgs[8],   img9=imgs[9],   img10=imgs[10], img11=imgs[11],
            img12=imgs[12], img13=imgs[13], img14=imgs[14], img15=imgs[15],
            img16=imgs[16], img17=imgs[17], img18=imgs[18], img19=imgs[19],
            img20=imgs[20], img21=imgs[21], img22=imgs[22], img23=imgs[23],
            img24=imgs[24], img25=imgs[25], img26=imgs[26], img27=imgs[27],
            img28=imgs[28], img29=imgs[29], img30=imgs[30], img31=imgs[31])
        imgs[0] = imgs_T['image']
        for i in range(1, 32):
            imgs[i] = imgs_T[f'img{i}']

        return imgs

    def transform_face(self, imgs):
        imgs_T = self.T(image=imgs[0], img1=imgs[1], img2=imgs[2], img3=imgs[3])
        imgs[0] = imgs_T['image']
        for i in range(1, 4):
            imgs[i] = imgs_T[f'img{i}']

        return imgs


class MHHRIDatasetS(Dataset):
    """MHHRI dataset, used for training from scratch."""

    def __init__(self, body_train_files, face_train_files, labels, transform=None):
        self.body_train_files = body_train_files
        self.face_train_files = face_train_files
        self.labels = labels
        self.T = transform

    def __getitem__(self, index):
        body_file = self.body_train_files[index]
        face_file = self.face_train_files[index]

        body = np.load(f'data/hhi_kinect_body_np/{body_file}').astype('float32')
        face = np.load(f'data/hhi_ego_face_np/{face_file}').astype('float32')
        label = self.labels[body_file]
        # print(np.shape(body), np.shape(face))

        if self.T is not None:
            body = self.transform_body(body)
            face = self.transform_face(face)

        # for i in range(32):
        #     cv2.imwrite(f'body_{i}.png', body[i])
        # for i in range(4):
        #     cv2.imwrite(f'face_{i}.png', face[i])

        return body, face, label

    def __len__(self):
        return len(self.body_train_files)

    def transform_body(self, imgs):
        imgs_T = self.T(
            image=imgs[0],  img1=imgs[1],   img2=imgs[2],   img3=imgs[3],
            img4=imgs[4],   img5=imgs[5],   img6=imgs[6],   img7=imgs[7],
            img8=imgs[8],   img9=imgs[9],   img10=imgs[10], img11=imgs[11],
            img12=imgs[12], img13=imgs[13], img14=imgs[14], img15=imgs[15],
            img16=imgs[16], img17=imgs[17], img18=imgs[18], img19=imgs[19],
            img20=imgs[20], img21=imgs[21], img22=imgs[22], img23=imgs[23],
            img24=imgs[24], img25=imgs[25], img26=imgs[26], img27=imgs[27],
            img28=imgs[28], img29=imgs[29], img30=imgs[30], img31=imgs[31])
        imgs[0] = imgs_T['image']
        for i in range(1, 32):
            imgs[i] = imgs_T[f'img{i}']

        return imgs

    def transform_face(self, imgs):
        imgs_T = self.T(image=imgs[0], img1=imgs[1], img2=imgs[2], img3=imgs[3])
        imgs[0] = imgs_T['image']
        for i in range(1, 4):
            imgs[i] = imgs_T[f'img{i}']

        return imgs


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
    # --------------------------------------------------------------------------
    # select ground truth file: [class, reg], [acq, self]
    label_path = f'data/annotations/{task}_{label_type}_session_norm.csv'
    labels = pd.read_csv(label_path)
    train_label = create_labels(self_body_train_files, labels, trait)
    test_label = create_labels(self_body_test_files, labels, trait)

    # --------------------------------------------------------------------------
    # data augmentation
    transform_train = A.Compose([
        # A.ToFloat(max_value=255),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(width=224, height=224, scale=[0.75, 1.0], p=0.5),
        # A.RandomBrightnessContrast(p=0.5)),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ], additional_targets={f'img{i}': 'image' for i in range(1, 32)}
    )
    # transform_test = A.Compose([
    #     # A.ToFloat(max_value=255),
    #     # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #     # ToTensorV2()
    #     ], additional_targets={f'img{i}': 'image' for i in range(1, 32)}
    # )

    # --------------------------------------------------------------------------
    train_data = MHHRIDataset(self_body_train_files, self_face_train_files,
        interact_body_train_files, interact_face_train_files,
        train_label, transform=transform_train)
    test_data = MHHRIDataset(self_body_test_files, self_face_test_files,
        interact_body_test_files, interact_face_test_files,
        test_label, transform=None)

    return train_data, test_data


def create_mhhri_s(body_train_files, body_test_files,
                  face_train_files, face_test_files,
                  task, label_type, trait):
    # --------------------------------------------------------------------------
    # select ground truth file: [class, reg], [acq, self]
    label_path = f'data/annotations/{task}_{label_type}_session_norm.csv'
    labels = pd.read_csv(label_path)
    train_label = create_labels(body_train_files, labels, trait)
    test_label = create_labels(body_test_files, labels, trait)

    # --------------------------------------------------------------------------
    # data augmentation
    transform_train = A.Compose([
        # A.ToFloat(max_value=255),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(width=224, height=224, scale=[0.75, 1.0], p=0.5),
        # A.RandomBrightnessContrast(p=0.5)),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ], additional_targets={f'img{i}': 'image' for i in range(1, 32)}
    )
    # transform_test = A.Compose([
    #     # A.ToFloat(max_value=255),
    #     # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #     # ToTensorV2()
    #     ], additional_targets={f'img{i}': 'image' for i in range(1, 32)}
    # )

    # --------------------------------------------------------------------------
    train_data = MHHRIDatasetS(body_train_files, face_train_files,
        train_label, transform=transform_train)
    test_data = MHHRIDatasetS(body_test_files, face_test_files,
        test_label, transform=None)

    return train_data, test_data


def split_filelists(s_body_list, s_face_list, i_body_list, i_face_list, fold):
    s_body_test = s_body_list[fold]
    s_body_train_list = np.delete(s_body_list, fold, axis=0)
    s_body_train = [item for row in s_body_train_list for item in row]
    # print(f'fold {fold}: train_num={len(s_body_train)}, test_num={len(s_body_test)}')

    s_face_test = s_face_list[fold]
    s_face_train_list = np.delete(s_face_list, fold, axis=0)
    s_face_train = [item for row in s_face_train_list for item in row]

    i_body_test = i_body_list[fold]
    i_body_train_list = np.delete(i_body_list, fold, axis=0)
    i_body_train = [item for row in i_body_train_list for item in row]

    i_face_test = i_face_list[fold]
    i_face_train_list = np.delete(i_face_list, fold, axis=0)
    i_face_train = [item for row in i_face_train_list for item in row]

    return s_body_train, s_body_test, s_face_train, s_face_test, i_body_train, i_body_test, i_face_train, i_face_test


def convert_image_to_npy(body_path_in, face_path_in, body_path_out, face_path_out):
    # set chunk size
    chunk_size_body = 32
    chunk_size_face = 4

    # sample body images --> 32 frames
    sessions = natsorted(os.listdir(body_path_in))
    persons = ['1', '2']
    for session in sessions:
        for person in persons:
            image_folder = body_path_in + session + '/' + person + '/'
            images = natsorted([i for i in os.listdir(image_folder)])
            frame_num = len(images)
            clip_num = frame_num // chunk_size_body
            frame_num = clip_num * chunk_size_body
            images = images[:frame_num]
            print(f'processing {session} {person}: {clip_num}')

            for i, clip_idx in enumerate(range(0, frame_num, chunk_size_body)):
                image_sequence = images[clip_idx:clip_idx+chunk_size_body]
                # print(image_sequence)
                clip_data = np.zeros((chunk_size_body, 224, 224, 3), dtype=np.int16)
                for frame_idx in range(chunk_size_body):
                    clip_data[frame_idx, :, :, :] = cv2.imread(
                        image_folder + image_sequence[frame_idx]
                    )
                file_name = f'{session}_{person}_clip{i+1}'
                np.save(f'{body_path_out}{file_name}', clip_data)

    # sample face images --> 4 frames
    sessions = natsorted(os.listdir(face_path_in))
    persons = ['1', '2']
    for session in sessions:
        for person in persons:
            image_folder = face_path_in + session + '/' + person + '/'
            images = natsorted([i for i in os.listdir(image_folder)])
            frame_num = len(images)
            clip_num = frame_num // chunk_size_face
            frame_num = clip_num * chunk_size_face
            images = images[:frame_num]
            print(f'processing {session} {person}: {clip_num}')

            for i, clip_idx in enumerate(range(0, frame_num, chunk_size_face)):
                image_sequence = images[clip_idx:clip_idx+chunk_size_face]
                # print(image_sequence)
                clip_data = np.zeros((chunk_size_face, 224, 224, 3), dtype=np.int16)
                for frame_idx in range(chunk_size_face):
                    clip_data[frame_idx, :, :, :] = cv2.imread(
                        image_folder + image_sequence[frame_idx]
                    )
                file_name = f'{session}_{person}_clip{i+1}'
                np.save(f'{face_path_out}{file_name}', clip_data)


if __name__ == '__main__':
    # --------------------------------------------------------------------------
    # convert body and face images to .npy file, each file represents a clip
    #   - body (32, 224, 224, 3), face (4, 224, 224, 3)
    #   - previous: r3d feature (1024, 4, 14, 14), face feature (512,)
    # r3d  : [b, 3, 32, 224, 224]
    # dmue : [b, 3, 224, 224]
    body_img_path = 'data/hhi_kinect_session_cropped/'
    face_img_path = 'data/hhi_ego_face_fixed/'
    body_npy_path = 'data/hhi_kinect_body_np/'
    face_npy_path = 'data/hhi_ego_face_np/'
    # convert_image_to_npy(body_img_path, face_img_path, body_npy_path, face_npy_path)

    # --------------------------------------------------------------------------
    fold_num = 6
    self_body_data_list = np.load('data/data_list/acq_self_body.npy', allow_pickle=True)
    self_face_data_list = np.load('data/data_list/acq_self_face.npy', allow_pickle=True)
    interact_body_data_list = np.load('data/data_list/acq_interact_body.npy', allow_pickle=True)
    interact_face_data_list = np.load('data/data_list/acq_interact_face.npy', allow_pickle=True)

    task = 'acq'
    label_type = 'reg'
    traits = ['O', 'C', 'E', 'A', 'N']

    for fold in range(fold_num):
        s_body_train, s_body_test, s_face_train, s_face_test, i_body_train, i_body_test, i_face_train, i_face_test = split_filelists(self_body_data_list, self_face_data_list, interact_body_data_list, interact_face_data_list, fold)

        # ----------------------------------------------------------------------
        # use self and interant data
        # train_data, test_data = create_mhhri(
        #     s_body_train, s_body_test, s_face_train, s_face_test,
        #     i_body_train, i_body_test, i_face_train, i_face_test,
        #     task=task, label_type=label_type, trait='O'
        # )

        # train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
        # test_loader = DataLoader(dataset=test_data, batch_size=16, shuffle=True)

        # for s_body_batch, s_face_batch, i_body_batch, i_face_batch, y_batch in train_loader:
        #     # s_body_batch  [16, 32, 224, 224, 3]  [16, 3, 32, 224, 224]
        #     # s_face_batch  [16,  4, 224, 224, 3]  [64, 3, 224, 224]
        #     # i_body_batch  [16, 32, 224, 224, 3]  [16, 3, 32, 224, 224]
        #     # i_face_batch  [16,  4, 224, 224, 3]  [64, 3, 224, 224]
        #     s_body_batch = s_body_batch.permute(0, 4, 1, 2, 3)
        #     s_face_batch = s_face_batch.permute(0, 1, 4, 2, 3)
        #     s_face_batch = s_face_batch.reshape(-1, 3, 224, 224)

        #     i_body_batch = i_body_batch.permute(0, 4, 1, 2, 3)
        #     i_face_batch = i_face_batch.permute(0, 1, 4, 2, 3)
        #     i_face_batch = i_face_batch.reshape(-1, 3, 224, 224)

        #     print(s_body_batch.size(), s_face_batch.size(), i_body_batch.size(), i_face_batch.size())
        #     break

        # break

        # ----------------------------------------------------------------------
        # use only self data
        train_data, test_data = create_mhhri_s(
            s_body_train, s_body_test, s_face_train, s_face_test,
            task=task, label_type=label_type, trait='O'
        )

        train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dataset=test_data, batch_size=16, shuffle=True)

        for body_batch, face_batch, y_batch in train_loader:
            # body_batch  [16, 32, 224, 224, 3]  [16, 3, 32, 224, 224]
            # face_batch  [16,  4, 224, 224, 3]  [64, 3, 224, 224]
            body_batch = body_batch.permute(0, 4, 1, 2, 3)
            face_batch = face_batch.permute(0, 1, 4, 2, 3)
            face_batch = face_batch.reshape(-1, 3, 224, 224)

            print(body_batch.size(), face_batch.size())
            break
        
        break
