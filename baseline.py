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

# from utils.prepare_data_bf_ibf import create_mhhri
from utils.mhhri import split_filelists, create_mhhri



def save_results(res, fold, output_path, trait):
    os.makedirs(f'{output_path}/csv/', exist_ok=True)

    cols = [
        'epoch', 'train_loss', 'test_loss', 'acc_r', 'r2', 'acc_c', 'bal_acc',
        'precision', 'recall', 'f1', 'auc'
    ]

    res = pd.DataFrame(res, columns=cols)
    res.to_csv(f'{output_path}/csv/{trait}_fold{fold}.csv', index=False)


def evaluate_res(y_true, y_pred, trait, fold, output_path):
    # save csv results
    os.makedirs(f'{output_path}/pred/', exist_ok=True)
    y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)
    y_out = pd.DataFrame(columns=['y_true', 'y_pred'])
    y_out['y_true'], y_out['y_pred'] = y_true, y_pred
    y_out.to_csv(f'{output_path}/pred/{trait}_fold{fold}.csv', index=False)

    # evaluate regression results
    acc_r = 1 - np.sum(np.abs(y_true - y_pred)) / len(y_true)
    r2 = 1 - np.sum((y_true - y_pred)**2) / len(y_true)

    # evaluate classification results
    y_true_c = np.where(y_true >= 0.5, 1, 0)
    y_pred_c = np.where(y_pred >= 0.5, 1, 0)

    acc_c = metrics.accuracy_score(y_true_c, y_pred_c)
    bal_acc = metrics.balanced_accuracy_score(y_true_c, y_pred_c)
    p, r, f1 = metrics.precision_recall_fscore_support(y_true_c,
                                                       y_pred_c,
                                                       average='macro')[:-1]
    fpr, tpr, _ = metrics.roc_curve(y_true_c, y_pred_c)
    auc = metrics.auc(fpr, tpr)

    return acc_r, r2, acc_c, bal_acc, p, r, f1, auc


def train(task, label_type, trait, output_path):
    fold_num = 6
    self_body_data_list = np.load('data/data_list/acq_self_body.npy', allow_pickle=True)
    self_face_data_list = np.load('data/data_list/acq_self_face.npy', allow_pickle=True)
    interact_body_data_list = np.load('data/data_list/acq_interact_body.npy', allow_pickle=True)
    interact_face_data_list = np.load('data/data_list/acq_interact_face.npy', allow_pickle=True)

    res_overall = {}
    for fold in range(fold_num):
        s_body_train, s_body_test, s_face_train, s_face_test, i_body_train, i_body_test, i_face_train, i_face_test = split_filelists(
            self_body_data_list, self_face_data_list, interact_body_data_list,
            interact_face_data_list, fold)

        train_data, test_data = create_mhhri(s_body_train, s_body_test,
                                             s_face_train, s_face_test,
                                             i_body_train, i_body_test,
                                             i_face_train, i_face_test, task,
                                             'reg', trait)

        train_loader = DataLoader(dataset=train_data,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS)
        test_loader = DataLoader(dataset=test_data,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True,
                                 num_workers=NUM_WORKERS)

        criterion = nn.MSELoss()

        logger.info(
            f'fold    | train_l  test_l |    acc     r2 |    '
            f'acc  b_acc      p      r     f1    auc'
        )
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
            acc_r, r2, acc_c, bal_acc, p, r, f1, auc = evaluate_res(
                y_true, y_pred, trait, fold, output_path)

            # logs
            logger.info(
                f'fold {fold} | '
                f'{train_loss.item(): 2.4f} {mean_loss.item(): 2.4f} | '
                f'{acc_r:.4f} {r2:.4f} | '
                f'{acc_c:.4f} {bal_acc:.4f} {p:.4f} {r:.4f} {f1:.4f} {auc:.4f}'
            )
            res.append([
                epoch,
                train_loss.item(),
                mean_loss.item(), acc_r, r2, acc_c, bal_acc, p, r, f1, auc
            ])
            res_overall[fold] = [acc_r, r2, acc_c, bal_acc, p, r, f1, auc]

        save_results(res, fold, output_path, trait)

    mean_acc_r, mean_r2 = 0.0, 0.0
    mean_acc_c, mean_bal_acc, mean_p, mean_r, mean_f1, mean_auc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for _, value in res_overall.items():
        mean_acc_r += value[0]
        mean_r2 += value[1]
        mean_acc_c += value[0]
        mean_bal_acc += value[1]
        mean_p += value[2]
        mean_r += value[3]
        mean_f1 += value[4]
        mean_auc += value[5]
    mean_acc_r = mean_acc_r / fold_num
    mean_r2 = mean_r2 / fold_num
    mean_acc_c = mean_acc_c / fold_num
    mean_bal_acc = mean_bal_acc / fold_num
    mean_p = mean_p / fold_num
    mean_r = mean_r / fold_num
    mean_f1 = mean_f1 / fold_num
    mean_auc = mean_auc / fold_num
    logger.info(
        f'                             avg | '
        f'{mean_acc_r:.4f} {mean_r2:.4f} | '
        f'{mean_acc_c:.4f} {mean_bal_acc:.4f} {mean_p:.4f} {mean_r:.4f} '
        f'{mean_f1:.4f} {mean_auc:.4f}'
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

    EPOCHS = 1
    BATCH_SIZE = 16
    NUM_WORKERS = 2

    for trait in traits:
        logger.info(
            f'{"=========="*8}\n'
            f'Configuration:'
            f'\n  TASK          : {task}'
            f'\n  TRAIT         : {trait}'
            f'\n  BATCH SIZE    : {BATCH_SIZE}'
            f'\n  EPOCH         : {EPOCHS}'
        )
        train(task, label_type, trait, output_path)
