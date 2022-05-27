import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from natsort import natsorted
from torch import nn


def visualize_acquitance_label():
    label_path = 'data/annotations/acquitance_label.csv'
    labels = pd.read_csv(label_path)
    print(labels.describe())

    print(labels['O'].value_counts())
    print(labels['C'].value_counts())
    print(labels['E'].value_counts())
    print(labels['A'].value_counts())
    print(labels['N'].value_counts())
    #     O   C   E   A   N
    # 0   9  11  10   9  10 
    # 1   9   7   8   9   8


def data_distribution_in_folds():
    label_path = 'data/annotations/acquitance_label_session.csv'
    labels = pd.read_csv(label_path)

    # folds = [['S01_kinect_1', 'S02_kinect_2', 'S11_kinect_2', 'S12_kinect_1'],
    #          ['S02_kinect_1', 'S04_kinect_1', 'S10_kinect_2', 'S07_kinect_2'],
    #          ['S01_kinect_2', 'S05_kinect_2', 'S07_kinect_1', 'S10_kinect_1'],
    #          ['S03_kinect_2', 'S11_kinect_1', 'S05_kinect_1', 'S08_kinect_1'],
    #          ['S03_kinect_1', 'S06_kinect_2', 'S09_kinect_2', 'S08_kinect_2'],
    #          ['S04_kinect_2', 'S06_kinect_1', 'S09_kinect_1', 'S12_kinect_2']]

    folds = [['S01_kinect_1', 'S02_kinect_2', 'S03_kinect_1', 'S06_kinect_1'],
             ['S01_kinect_2', 'S02_kinect_1', 'S11_kinect_2', 'S12_kinect_1'],
             ['S03_kinect_2', 'S05_kinect_1', 'S07_kinect_1', 'S12_kinect_2'],
             ['S04_kinect_1', 'S07_kinect_2', 'S09_kinect_1', 'S10_kinect_1'],
             ['S04_kinect_2', 'S05_kinect_2', 'S08_kinect_1', 'S10_kinect_2'],
             ['S06_kinect_2', 'S08_kinect_2', 'S09_kinect_2', 'S11_kinect_1']]

    for fold_idx, fold in enumerate(folds):
        print(f'fold {fold_idx+1}')
        test_list = fold
        # print(test_list)
        test_label = pd.DataFrame()
        for test_item in test_list:
            test_label = test_label.append(labels[labels['session'] == test_item], ignore_index=True)
        # print(test_label)
        
        train_list = np.delete(folds, fold_idx, axis=0)
        train_list = [item for row in train_list for item in row]
        # print(train_list)
        train_label = pd.DataFrame()
        for train_item in train_list:
            train_label = train_label.append(labels[labels['session'] == train_item], ignore_index=True)
        # print(train_label)

        print('train')
        count_O = np.array(np.unique(np.array(train_label['O']), return_counts=True)).astype(int)[1]
        count_C = np.array(np.unique(np.array(train_label['C']), return_counts=True)).astype(int)[1]
        count_E = np.array(np.unique(np.array(train_label['E']), return_counts=True)).astype(int)[1]
        count_A = np.array(np.unique(np.array(train_label['A']), return_counts=True)).astype(int)[1]
        count_N = np.array(np.unique(np.array(train_label['N']), return_counts=True)).astype(int)[1]

        print(count_O)
        print(count_C)
        print(count_E)
        print(count_A)
        print(count_N)

        print('test')
        count_O = np.array(np.unique(np.array(test_label['O']), return_counts=True)).astype(int)[1]
        count_C = np.array(np.unique(np.array(test_label['C']), return_counts=True)).astype(int)[1]
        count_E = np.array(np.unique(np.array(test_label['E']), return_counts=True)).astype(int)[1]
        count_A = np.array(np.unique(np.array(test_label['A']), return_counts=True)).astype(int)[1]
        count_N = np.array(np.unique(np.array(test_label['N']), return_counts=True)).astype(int)[1]

        print(count_O)
        print(count_C)
        print(count_E)
        print(count_A)
        print(count_N)

        print()


def view_label():
    acq = pd.read_csv('data/annotations/acq_reg.csv')
    self = pd.read_csv('data/annotations/self_reg.csv')
    x = ['O', 'C', 'E', 'A', 'N']
    fig, (ax1, ax2) = plt.subplots(figsize=(8, 6), dpi=80, ncols=2)

    y = [self.iloc[:,-1], self.iloc[:,-3], self.iloc[:,-5], self.iloc[:,-4], self.iloc[:,-2]]
    for xe, ye in zip(x, y):
        ax1.scatter([xe] * len(ye), ye)
    ax1.title.set_text('True Personality')
    ax1.grid()
    ax1.set_ylim([0.5, 10.5])
    ax1.set_yticks(range(1, 11))

    y = [acq.iloc[:,-1], acq.iloc[:,-3], acq.iloc[:,-5], acq.iloc[:,-4], acq.iloc[:,-2]]
    for xe, ye in zip(x, y):
        ax2.scatter([xe] * len(ye), ye)
    ax2.title.set_text('Apparent Personality')
    ax2.grid()
    ax2.set_ylim([0.5, 10.5])
    ax2.set_yticks(range(1, 11))

    plt.show()
    fig.savefig(f'visualize/label/label.png')


view_label()
