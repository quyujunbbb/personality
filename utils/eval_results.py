import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from natsort import natsorted
from sklearn import metrics
from torch.utils.data import DataLoader


def calculate_average_ocean():
    # results_acq_class, results_acq_reg, results_self_class, results_self_reg
    res_file_path = 'results/results_acq_reg.csv'
    res = pd.read_csv(res_file_path, sep=',')

    for row in range(len(res)):
        ocean = np.array([
            res.iloc[row, -6],
            res.iloc[row, -5],
            res.iloc[row, -4],
            res.iloc[row, -3],
            res.iloc[row, -2],
        ])
        res.iloc[row, -1] = np.round(np.average(ocean), 4)

    print(res)


def average_array():
    a = np.array([[0.9016, 0.8616, 0.8453, 0.8511, 0.8190],
                  [0.9858, 0.9736, 0.9577, 0.9615, 0.9553],
                  [0.9072, 0.8751, 0.8470, 0.8670, 0.8292],
                  [0.9843, 0.9750, 0.9578, 0.9718, 0.9460]])
    for row in range(len(a)):
        print(np.average(a[row]))


if __name__ == '__main__':
    # calculate_average_ocean()
    average_array()
