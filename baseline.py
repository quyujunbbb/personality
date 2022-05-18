import time
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn import metrics
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from utils.prepare_data_bf_ibf import create_mhhri



def save_results(res, fold, output_path, trait):
    csv_path = f'{output_path}/csv/'
    fig_path = f'{output_path}/figs/'
    os.makedirs(csv_path, exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)

    clms = ['epoch', 'train_loss', 'test_loss', 'acc', 'r2']
    res = pd.DataFrame(res, columns=clms)
    res.to_csv(csv_path + f'{trait}_fold{fold}.csv', index=False)

    fig, ax = plt.subplots()
    ax.plot(res['epoch'], res['train_loss'], label='train loss')
    ax.plot(res['epoch'], res['test_loss'], label='test loss')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.grid()
    ax.legend()
    fig.savefig(fig_path + f'{trait}_fold{fold}_loss.png')

    fig, ax = plt.subplots()
    ax.plot(res['epoch'], res['acc'], label='test accuray')
    ax.set_xlabel('epochs')
    ax.set_ylabel('test accuray')
    ax.grid()
    ax.legend()
    fig.savefig(fig_path + f'{trait}_fold{fold}_acc.png')

    smooth_factor = 0.8
    smooth_train = res['train_loss'].ewm(alpha=(1 - smooth_factor)).mean()
    smooth_test = res['test_loss'].ewm(alpha=(1 - smooth_factor)).mean()

    fig, ax = plt.subplots()
    ax.plot(res['epoch'], smooth_train, label='train loss')
    ax.plot(res['epoch'], smooth_test, label='test loss')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.grid()
    ax.legend()
    fig.savefig(fig_path + f'{trait}_fold{fold}_loss_s.png')


def evaluate_res_reg(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = np.sum(1 - np.abs(y_true - y_pred)) / len(y_true)
    r2 = 1 - np.sum((y_true - y_pred)**2) / len(y_true)

    return acc, r2


def train(task, label_type, trait, output_path):
    fold_num = 6
    self_body_data_list = np.load('data/data_list/acq_self_body.npy', allow_pickle=True)
    self_face_data_list = np.load('data/data_list/acq_self_face.npy', allow_pickle=True)
    interact_body_data_list = np.load('data/data_list/acq_interact_body.npy', allow_pickle=True)
    interact_face_data_list = np.load('data/data_list/acq_interact_face.npy', allow_pickle=True)

    res_overall = {}
    for fold in range(fold_num):
        self_body_test_files = self_body_data_list[fold]
        self_body_train_list = np.delete(self_body_data_list, fold, axis=0)
        self_body_train_files = [item for row in self_body_train_list for item in row]

        self_face_test_files = self_face_data_list[fold]
        self_face_train_list = np.delete(self_face_data_list, fold, axis=0)
        self_face_train_files = [item for row in self_face_train_list for item in row]

        interact_body_test_files = interact_body_data_list[fold]
        interact_body_train_list = np.delete(interact_body_data_list, fold, axis=0)
        interact_body_train_files = [item for row in interact_body_train_list for item in row]

        interact_face_test_files = interact_face_data_list[fold]
        interact_face_train_list = np.delete(interact_face_data_list, fold, axis=0)
        interact_face_train_files = [item for row in interact_face_train_list for item in row]
        
        logger.info(
            f'fold {fold} - train_num={len(self_body_train_files)}, test_num={len(self_body_test_files)}'
        )

        train_data, test_data = create_mhhri(
            self_body_train_files, self_body_test_files,
            self_face_train_files, self_face_test_files,
            interact_body_train_files, interact_body_test_files,
            interact_face_train_files, interact_face_test_files,
            task, label_type, trait
        )

        train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

        criterion = nn.MSELoss()

        logger.info('epoch   train_l  test_l     acc      r2')
        res = []
        for epoch in range(EPOCHS):
            # train
            for x_self_body_batch, x_self_face_batch, x_interact_body_batch, x_interact_face_batch, y_batch in train_loader:
                if torch.cuda.is_available():
                    x_self_body_batch = x_self_body_batch
                    x_self_face_batch = x_self_face_batch
                    x_interact_body_batch = x_interact_body_batch
                    x_interact_face_batch = x_interact_face_batch
                    y_batch = y_batch.clone().detach().to(torch.float).view(-1,1)

                y_out = torch.rand(y_batch.size())
                train_loss = criterion(y_out, y_batch)

            # test
            total_loss = []
            true_label_list = []
            pred_label_list = []

            for x_self_body_batch, x_self_face_batch, x_interact_body_batch, x_interact_face_batch, y_batch in test_loader:
                if torch.cuda.is_available():
                    x_self_body_batch = x_self_body_batch
                    x_self_face_batch = x_self_face_batch
                    x_interact_body_batch = x_interact_body_batch
                    x_interact_face_batch = x_interact_face_batch
                    y_batch = y_batch.clone().detach().to(torch.float).view(-1,1)

                y_out = torch.rand(y_batch.size())
                test_loss = criterion(y_out, y_batch)
                total_loss.append(test_loss)

                true_label_list.append(y_batch.cpu().detach().numpy())
                pred_label_list.append(y_out.cpu().detach().numpy())

            mean_loss = sum(total_loss) / total_loss.__len__()
            y_true = np.concatenate(true_label_list)
            y_pred = np.concatenate(pred_label_list)
            acc, r2 = evaluate_res_reg(y_true, y_pred)

            # logs
            logger.info(
                f'[{epoch+1:02d}/{EPOCHS:02d}] '
                f'{train_loss.item(): 2.4f} {mean_loss.item(): 2.4f} {acc: 2.4f} {r2: 2.4f}')
            res.append([epoch, train_loss.item(), mean_loss.item(), acc, r2])
            res_overall[fold] = [acc, r2]

        save_results(res, fold, output_path, trait)

    mean_acc, mean_r2 = 0.0, 0.0
    for _, value in res_overall.items():
        mean_acc += value[0]
        mean_r2 += value[1]
    mean_acc = mean_acc / fold_num
    mean_r2 = mean_r2 / fold_num
    logger.info(
        f'Average: acc={mean_acc:.4f} b_acc={mean_r2:.4f}\n'
    )


if __name__ == '__main__':
    # torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    output_path = f'results/baseline_reg/'
    os.makedirs(output_path, exist_ok=True)
    logger.add(f'{output_path}/log.txt', format='{message}', level='INFO')

    task = 'acq'  # or 'self'
    label_type = 'reg'  # 'class'
    traits = ['O', 'C', 'E', 'A', 'N']

    EPOCHS = 10
    BATCH_SIZE = 32

    for trait in traits:
        logger.info(
            f'{"=========="*8}\n'
            f'Configuration:'
            f'\n  TASK          : {task}'
            f'\n  TRAIT         : {trait}'
            f'\n  BATCH SIZE    : {BATCH_SIZE}'
            f'\n  EPOCH         : {EPOCHS}\n'
        )
        train(task, label_type, trait, output_path)
