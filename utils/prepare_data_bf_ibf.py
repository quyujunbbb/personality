import os
import random

import numpy as np
import pandas as pd
from natsort import natsorted
from torch.utils.data import Dataset


class MHHRI(Dataset):

    def __init__(self, self_body_train_files, self_face_train_files,
                 interact_body_train_files, interact_face_train_files, labels):
        self.self_body_train_files = self_body_train_files
        self.self_face_train_files = self_face_train_files
        self.interact_body_train_files = interact_body_train_files
        self.interact_face_train_files = interact_face_train_files
        self.labels = labels

    def __getitem__(self, index):
        self_body_feature_file = self.self_body_train_files[index]
        self_face_feature_file = self.self_face_train_files[index]
        interact_body_feature_file = self.interact_body_train_files[index]
        interact_face_feature_file = self.interact_face_train_files[index]

        self_body_feature = np.load(f'features/r3d_features/{self_body_feature_file}')
        self_face_feature = np.load(f'features/face_features_fixed/{self_face_feature_file}')
        # self_face_feature = np.load(f'features/face_features/{self_face_feature_file}')

        interact_body_feature = np.load(f'features/r3d_features/{interact_body_feature_file}')
        interact_face_feature = np.load(f'features/face_features_fixed/{interact_face_feature_file}')
        # interact_face_feature = np.load(f'features/face_features/{interact_face_feature_file}')

        label = self.labels[self_body_feature_file]

        return self_body_feature, self_face_feature, interact_body_feature, interact_face_feature, label

    def __len__(self):
        return len(self.self_body_train_files)


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

    # folder split method 2 - self
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

    self_body_data_list = np.array([list_1, list_2, list_3, list_4, list_5, list_6], dtype=object)
    np.save('data/data_list/acq_self_body', self_body_data_list)

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

    self_face_data_list = np.array([list_1, list_2, list_3, list_4, list_5, list_6], dtype=object)
    np.save('data/data_list/acq_self_face', self_face_data_list)

    # folder split method 2 - interactant
    fold_1 = ['S01_2', 'S02_1', 'S03_2', 'S06_2']
    fold_2 = ['S01_1', 'S02_2', 'S11_1', 'S12_2']
    fold_3 = ['S03_1', 'S05_2', 'S07_2', 'S12_1']
    fold_4 = ['S04_2', 'S07_1', 'S09_2', 'S10_2']
    fold_5 = ['S04_1', 'S05_1', 'S08_2', 'S10_1']
    fold_6 = ['S06_1', 'S08_1', 'S09_1', 'S11_2']

    # --------------------------------------------------------------------------
    # interactant body
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

    interact_body_data_list = np.array([list_1, list_2, list_3, list_4, list_5, list_6], dtype=object)
    np.save('data/data_list/acq_interact_body', interact_body_data_list)

    # --------------------------------------------------------------------------
    # interactant face
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

    interact_face_data_list = np.array([list_1, list_2, list_3, list_4, list_5, list_6], dtype=object)
    np.save('data/data_list/acq_interact_face', interact_face_data_list)

    return self_body_data_list, self_face_data_list, interact_body_data_list, interact_face_data_list


if __name__ == '__main__':
    # == Option 1 ==
    # generate data_list from beginning
    # body_data_path = 'features/r3d_features/'
    # face_data_path = 'features/face_features_fixed/'
    # self_body_data_list, self_face_data_list, interact_body_data_list, interact_face_data_list = create_data_list(body_data_path, face_data_path)

    # print(self_body_data_list[0][:20])
    # print(self_face_data_list[0][:20])
    # print(interact_body_data_list[0][:20])
    # print(interact_face_data_list[0][:20])

    # == Option 2 ==
    fold_num = 6
    self_body_data_list = np.load('data/data_list/acq_self_body.npy', allow_pickle=True)
    self_face_data_list = np.load('data/data_list/acq_self_face.npy', allow_pickle=True)
    interact_body_data_list = np.load('data/data_list/acq_interact_body.npy', allow_pickle=True)
    interact_face_data_list = np.load('data/data_list/acq_interact_face.npy', allow_pickle=True)

    # print(self_body_data_list[0][:20])
    # print(self_face_data_list[0][:20])
    # print(interact_body_data_list[0][:20])
    # print(interact_face_data_list[0][:20])

    task = 'acq'
    label_type = 'reg'
    traits = ['O', 'C', 'E', 'A', 'N']

    for fold in range(fold_num):
        self_body_test_files = self_body_data_list[fold]
        self_body_train_list = np.delete(self_body_data_list, fold, axis=0)
        self_body_train_files = [item for row in self_body_train_list for item in row]
        print(f'fold {fold}: train_num={len(self_body_train_files)}, test_num={len(self_body_test_files)}')

        self_face_test_files = self_face_data_list[fold]
        self_face_train_list = np.delete(self_face_data_list, fold, axis=0)
        self_face_train_files = [item for row in self_face_train_list for item in row]
        print(f'fold {fold}: train_num={len(self_face_train_files)}, test_num={len(self_face_test_files)}')

        interact_body_test_files = interact_body_data_list[fold]
        interact_body_train_list = np.delete(interact_body_data_list, fold, axis=0)
        interact_body_train_files = [item for row in interact_body_train_list for item in row]
        print(f'fold {fold}: train_num={len(interact_body_train_files)}, test_num={len(interact_body_test_files)}')

        interact_face_test_files = interact_face_data_list[fold]
        interact_face_train_list = np.delete(interact_face_data_list, fold, axis=0)
        interact_face_train_files = [item for row in interact_face_train_list for item in row]
        print(f'fold {fold}: train_num={len(interact_face_train_files)}, test_num={len(interact_face_test_files)}')

        for trait in traits:
            train_data, test_data = create_mhhri(
                self_body_train_files, self_body_test_files,
                self_face_train_files, self_face_test_files,
                interact_body_train_files,  interact_body_test_files,
                interact_face_train_files, interact_face_test_files,
                task=task, label_type=label_type, trait=trait
            )
            break
        break
