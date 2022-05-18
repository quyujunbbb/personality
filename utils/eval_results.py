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


if __name__ == '__main__':
    calculate_average_ocean()
