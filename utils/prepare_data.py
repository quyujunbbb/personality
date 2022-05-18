import os
import random

import numpy as np
import pandas as pd
from natsort import natsorted
from torch.utils.data import Dataset


class MHHRI(Dataset):

    def __init__(self, feature_files, labels):
        self.feature_files = feature_files
        self.labels = labels

    def __getitem__(self, index):
        feature_file = self.feature_files[index]
        feature = np.load(f'features/r3d_features_clip/{feature_file}')
        label = self.labels[feature_file]
        return feature, label

    def __len__(self):
        return len(self.feature_files)


def create_labels(files, labels, trait):
    labels = labels[['session', trait]]
    label_out = {}

    for file in files:
        file_name = file.split('.')[0]
        sess_name = f"{file_name.split('_')[0]}_{file_name.split('_')[1]}_{file_name.split('_')[2]}"
        label_out[file] = labels[labels['session'] == sess_name][trait].values[0]

    return label_out


def split_data_clip_level(data_path):
    data_files = natsorted(os.listdir(data_path))
    random.shuffle(data_files)

    fold_num = 6
    session_num = int(len(data_files) / 6)

    split_info = []
    for fold in range(fold_num):
        print(fold*session_num, (fold+1)*session_num)
        data_fold = data_files[fold*session_num:(fold+1)*session_num]
        split_info.append(data_fold)

    np.save('data/data_list/clip_level_list', split_info)


def split_data_session_user_level(data_path):
    data_files = natsorted(os.listdir(data_path))
    # print(len(data_files))

    fold_1 = ['S01_1', 'S02_2', 'S11_2', 'S12_1']
    fold_2 = ['S02_1', 'S04_1', 'S10_2', 'S07_2']
    fold_3 = ['S01_2', 'S05_2', 'S07_1', 'S10_1']
    fold_4 = ['S03_2', 'S11_1', 'S05_1', 'S08_1']
    fold_5 = ['S03_1', 'S06_2', 'S09_2', 'S08_2']
    fold_6 = ['S04_2', 'S06_1', 'S09_1', 'S12_2']

    list_1, list_2, list_3, list_4, list_5, list_6 = [], [], [], [], [], []
    for data_file in data_files:
        session, _, user, _ = data_file.split('.')[0].split('_')
        session_user = session + '_' + user
        if   session_user in fold_1: list_1.append(data_file)
        elif session_user in fold_2: list_2.append(data_file)
        elif session_user in fold_3: list_3.append(data_file)
        elif session_user in fold_4: list_4.append(data_file)
        elif session_user in fold_5: list_5.append(data_file)
        elif session_user in fold_6: list_6.append(data_file)

    # print(len(list_1), len(list_2), len(list_3), len(list_4), len(list_5), len(list_6))
    data_list = np.array([list_1, list_2, list_3, list_4, list_5, list_6], dtype=object)
    np.save('data/data_list/session_user_level_list', data_list)

    return data_list


def create_mhhri(train_files, test_files, trait):
    label_path = 'data/annotations/acq_class_session.csv'
    labels = pd.read_csv(label_path)

    train_label = create_labels(train_files, labels, trait)
    test_label = create_labels(test_files, labels, trait)

    train_data = MHHRI(train_files, train_label)
    test_data = MHHRI(test_files, test_label)

    return train_data, test_data


if __name__ == '__main__':
    # == Option 1 ==
    # generate data_list from beginning
    # data_path = 'features/r3d_features_clip/'
    # data_list = split_data_session_user_level(data_path)

    # == Option 2 ==
    fold_num = 6
    data_list = np.load('data/data_list/session_user_level_list.npy', allow_pickle=True)

    for fold in range(fold_num):
        test_files = data_list[fold]
        train_list = np.delete(data_list, fold, axis=0)
        train_files = [item for row in train_list for item in row]
        print(f'fold {fold}: train_num={len(train_files)}, test_num={len(test_files)}')

        create_mhhri(train_files, test_files, trait='E')
        # create_mhhri(train_files, test_files, trait='A')
        # create_mhhri(train_files, test_files, trait='C')
        # create_mhhri(train_files, test_files, trait='N')
        # create_mhhri(train_files, test_files, trait='O')
