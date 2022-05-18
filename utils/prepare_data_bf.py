import os
import random

import numpy as np
import pandas as pd
from natsort import natsorted
from torch.utils.data import Dataset


class MHHRI(Dataset):

    def __init__(self, body_feature_files, face_feature_files, labels):
        self.body_feature_files = body_feature_files
        self.face_feature_files = face_feature_files
        self.labels = labels

    def __getitem__(self, index):
        body_feature_file = self.body_feature_files[index]
        face_feature_file = self.face_feature_files[index]
        body_feature = np.load(f'features/r3d_features/{body_feature_file}')
        face_feature = np.load(f'features/face_features/{face_feature_file}')

        label = self.labels[body_feature_file]

        return body_feature, face_feature, label

    def __len__(self):
        return len(self.body_feature_files)


def create_labels(files, labels, trait):
    labels = labels[['session', trait]]
    label_out = {}

    for file in files:
        file_name = file.split('.')[0]
        sess_name = f"{file_name.split('_')[0]}_{file_name.split('_')[1]}_{file_name.split('_')[2]}"
        label_out[file] = labels[labels['session'] == sess_name][trait].values[0]

    return label_out


def create_mhhri(body_train_files, body_test_files, face_train_files, face_test_files, trait):
    # select acquitance or self label
    # label_path = 'data/annotations/acq_class_session.csv'
    label_path = 'data/annotations/self_class_session.csv'
    labels = pd.read_csv(label_path)

    train_label = create_labels(body_train_files, labels, trait)
    test_label = create_labels(body_test_files, labels, trait)
    # with open(f'{fold}_{trait}.csv', 'w') as f:
    #     for key in test_label.keys():
    #         f.write("%s, %s\n" % (key, test_label[key]))

    train_data = MHHRI(body_train_files, face_train_files, train_label)
    test_data = MHHRI(body_test_files, face_test_files, test_label)

    return train_data, test_data


def create_data_list(body_data_path, face_data_path):
    body_data_files = natsorted(os.listdir(body_data_path))
    face_data_files = natsorted(os.listdir(face_data_path))

    # folder split method 1
    # fold_1 = ['S01_1', 'S02_2', 'S11_2', 'S12_1']
    # fold_2 = ['S02_1', 'S04_1', 'S10_2', 'S07_2']
    # fold_3 = ['S01_2', 'S05_2', 'S07_1', 'S10_1']
    # fold_4 = ['S03_2', 'S11_1', 'S05_1', 'S08_1']
    # fold_5 = ['S03_1', 'S06_2', 'S09_2', 'S08_2']
    # fold_6 = ['S04_2', 'S06_1', 'S09_1', 'S12_2']

    # folder split method 2
    fold_1 = ['S01_1', 'S02_2', 'S03_1', 'S06_1']
    fold_2 = ['S01_2', 'S02_1', 'S11_2', 'S12_1']
    fold_3 = ['S03_2', 'S05_1', 'S07_1', 'S12_2']
    fold_4 = ['S04_1', 'S07_2', 'S09_1', 'S10_1']
    fold_5 = ['S04_2', 'S05_2', 'S08_1', 'S10_2']
    fold_6 = ['S06_2', 'S08_2', 'S09_2', 'S11_1']

    # --------------------------------------------------------------------------
    # self body
    list_1, list_2, list_3, list_4, list_5, list_6 = [], [], [], [], [], []
    for body_data_file in body_data_files:
        session, _, user, _ = body_data_file.split('.')[0].split('_')
        session_user = session + '_' + user
        if   session_user in fold_1: list_1.append(body_data_file)
        elif session_user in fold_2: list_2.append(body_data_file)
        elif session_user in fold_3: list_3.append(body_data_file)
        elif session_user in fold_4: list_4.append(body_data_file)
        elif session_user in fold_5: list_5.append(body_data_file)
        elif session_user in fold_6: list_6.append(body_data_file)
    print(len(list_1), len(list_2), len(list_3), len(list_4), len(list_5), len(list_6))

    body_data_list = np.array([list_1, list_2, list_3, list_4, list_5, list_6], dtype=object)
    np.save('data/data_list/session_user_level_body_list', body_data_list)

    # --------------------------------------------------------------------------
    # self face
    list_1, list_2, list_3, list_4, list_5, list_6 = [], [], [], [], [], []
    for face_data_file in face_data_files:
        session, _, user, _ = face_data_file.split('.')[0].split('_')
        session_user = session + '_' + user
        if   session_user in fold_1: list_1.append(face_data_file)
        elif session_user in fold_2: list_2.append(face_data_file)
        elif session_user in fold_3: list_3.append(face_data_file)
        elif session_user in fold_4: list_4.append(face_data_file)
        elif session_user in fold_5: list_5.append(face_data_file)
        elif session_user in fold_6: list_6.append(face_data_file)
    print(len(list_1), len(list_2), len(list_3), len(list_4), len(list_5), len(list_6))

    face_data_list = np.array([list_1, list_2, list_3, list_4, list_5, list_6], dtype=object)
    np.save('data/data_list/session_user_level_face_list', face_data_list)

    return body_data_list, face_data_list


if __name__ == '__main__':
    # == Option 1 ==
    # generate data_list from beginning
    # body_data_path = 'features/r3d_features/'
    # face_data_path = 'features/face_features/'
    # body_data_list, face_data_list = create_data_list(body_data_path, face_data_path)

    # == Option 2 ==
    fold_num = 6
    body_data_list = np.load('data/data_list/session_user_level_body_list_2.npy', allow_pickle=True)
    face_data_list = np.load('data/data_list/session_user_level_face_list_2.npy', allow_pickle=True)

    for fold in range(fold_num):
        body_test_files = body_data_list[fold]
        body_train_list = np.delete(body_data_list, fold, axis=0)
        body_train_files = [item for row in body_train_list for item in row]
        print(f'fold {fold}: train_num={len(body_train_files)}, test_num={len(body_test_files)}')

        face_test_files = face_data_list[fold]
        face_train_list = np.delete(face_data_list, fold, axis=0)
        face_train_files = [item for row in face_train_list for item in row]
        print(f'fold {fold}: train_num={len(face_train_files)}, test_num={len(face_test_files)}')

        train_data, test_data = create_mhhri(body_train_files, body_test_files, face_train_files, face_test_files, trait='O')
        train_data, test_data = create_mhhri(body_train_files, body_test_files, face_train_files, face_test_files, trait='C')
        train_data, test_data = create_mhhri(body_train_files, body_test_files, face_train_files, face_test_files, trait='E')
        train_data, test_data = create_mhhri(body_train_files, body_test_files, face_train_files, face_test_files, trait='A')
        train_data, test_data = create_mhhri(body_train_files, body_test_files, face_train_files, face_test_files, trait='N')
