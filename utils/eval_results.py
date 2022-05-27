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
    # a = np.array([[0.9348, 0.8478, 0.8355, 0.7058, 0.6898],
    #               [0.8135, 0.7026, 0.6123, 0.4034, 0.5101],
    #               [0.8497, 1.0000, 0.8198, 0.9633, 0.9854],
    #               [0.6662, 1.0000, 0.7019, 0.9064, 0.9128]])
    # for row in range(len(a)):
    #     print(np.average(a[row]))

    res = pd.read_csv('temp.csv')
    res['Avg_acc'] = np.round((res['O_acc'] + res['C_acc'] + res['E_acc'] + res['A_acc'] + res['N_acc']) / 5, 4)
    res['Avg_r2'] = np.round((res['O_r2'] + res['C_r2'] + res['E_r2'] + res['A_r2'] + res['N_r2']) / 5, 4)

    res_true = res.iloc[:2,:]
    res_appa = res.iloc[2:,:]
    
    print(res_true)
    print(res_appa)
    print(res)
    
    # res = pd.read_csv('temp.csv')
    # o = res.iloc[:6,:].mean()
    # c = res.iloc[6:12,:].mean()
    # e = res.iloc[12:18,:].mean()
    # a = res.iloc[18:24,:].mean()
    # n = res.iloc[24:30,:].mean()
    # print(o)
    # print(c)
    # print(e)
    # print(a)
    # print(n)


if __name__ == '__main__':
    # calculate_average_ocean()
    average_array()
