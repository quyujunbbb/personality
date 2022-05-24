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
    a = np.array([[0.8991, 0.8453, 0.8596, 0.8369, 0.8387],
                  [0.9848, 0.9668, 0.9640, 0.9610, 0.9515],
                  [0.9595, 0.9376, 0.8980, 0.9302, 0.9437],
                  [0.9977, 0.9906, 0.9836, 0.9880, 0.9944]])
    for row in range(len(a)):
        print(np.average(a[row]))

    0.8991, 0.8453, 0.8596, 0.8369, 0.8387
    0.9848, 0.9668, 0.9640, 0.9610, 0.9515
if __name__ == '__main__':
    # calculate_average_ocean()
    average_array()
