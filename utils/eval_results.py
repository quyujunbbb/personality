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
    a = np.array([[0.9210, 0.9289, 0.8871, 0.9378, 0.9270],
                  [0.9923, 0.9903, 0.9811, 0.9918, 0.9929],
                  [0.9536, 0.9313, 0.8994, 0.9273, 0.9435],
                  [0.9969, 0.9891, 0.9842, 0.9889, 0.9953],
                  [0.9039, 0.8826, 0.8377, 0.8576, 0.8415],
                  [0.9831, 0.9771, 0.9576, 0.9680, 0.9536],
                  [0.9120, 0.8797, 0.8449, 0.8578, 0.8399],
                  [0.9860, 0.9737, 0.9483, 0.9658, 0.9588]])

    for row in range(len(a)):
        print(np.average(a[row]))

    0.8991, 0.8453, 0.8596, 0.8369, 0.8387
    0.9848, 0.9668, 0.9640, 0.9610, 0.9515
if __name__ == '__main__':
    # calculate_average_ocean()
    average_array()
