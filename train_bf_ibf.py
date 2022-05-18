import argparse
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

from models import nets
from utils.prepare_data_bf_ibf import create_mhhri


def make_parser():
    # task  : acq, self
    # label : class, reg
    # trait : O, C, E, A, N, ALL
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', type=str)
    parser.add_argument('--task', dest='task', type=str)
    parser.add_argument('--label', dest='label_type', type=str)
    parser.add_argument('--trait', dest='trait', type=str)

    return parser


def save_results(res, fold, output_path, label_type, trait):
    csv_path = f'{output_path}/csv/'
    fig_path = f'{output_path}/figs/'
    os.makedirs(csv_path, exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)

    assert(label_type in ['class', 'reg'])
    if label_type == 'class':
        cols = ['epoch', 'train_loss', 'test_loss', 'acc', 'bal_acc',
                'precision', 'recall', 'f1', 'auc']
    elif label_type == 'reg':
        cols = ['epoch', 'train_loss', 'test_loss', 'acc', 'r2']

    res = pd.DataFrame(res, columns=cols)
    res.to_csv(csv_path + f'{trait}_fold{fold}.csv', index=False)

    # draw loss
    fig, ax = plt.subplots()
    ax.plot(res['epoch'], res['train_loss'], label='train loss')
    ax.plot(res['epoch'], res['test_loss'], label='test loss')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.grid()
    ax.legend()
    fig.savefig(fig_path + f'{trait}_fold{fold}_loss.png')

    # draw loss smooth
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

    # draw accuracy
    fig, ax = plt.subplots()
    ax.plot(res['epoch'], res['acc'], label='test accuray')
    ax.set_xlabel('epochs')
    ax.set_ylabel('test accuray')
    ax.grid()
    ax.legend()
    fig.savefig(fig_path + f'{trait}_fold{fold}_acc.png')


def evaluate_res(y_true, y_pred, label_type):
    assert(label_type in ['class', 'reg'])

    if label_type == 'class':
        # evaluate results
        acc = metrics.accuracy_score(y_true, y_pred)
        bal_acc = metrics.balanced_accuracy_score(y_true, y_pred)
        p, r, f1 = metrics.precision_recall_fscore_support(y_true, y_pred, average='macro')[:-1]
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
        auc = metrics.auc(fpr, tpr)

        # save csv results
        y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)
        y_out = pd.DataFrame(columns=['y_true', 'y_pred'])
        y_out['y_true'], y_out['y_pred'] = y_true, y_pred
        y_out.to_csv(f'y.csv', index=False)

        return acc, bal_acc, p, r, f1, auc

    elif label_type == 'reg':
        # evaluate results
        acc = 1 - np.sum(np.abs(y_true - y_pred)) / len(y_true)
        r2 = 1 - np.sum((y_true - y_pred)**2) / len(y_true)

        # save csv results
        y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)
        y_out = pd.DataFrame(columns=['y_true', 'y_pred'])
        y_out['y_true'], y_out['y_pred'] = y_true, y_pred
        y_out.to_csv(f'y.csv', index=False)

        return acc, r2


def train_reg(model, task, label_type, trait, timestamp, output_path):
    """Train model."""
    ckpt_save_path = f'{output_path}ckpts/'
    os.makedirs(ckpt_save_path, exist_ok=True)
    writer = SummaryWriter('runs')

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

        train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=NUM_WORKERS)
        test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE,
                                 shuffle=True, num_workers=NUM_WORKERS)

        net = model()
        if torch.cuda.is_available():
            net = nn.DataParallel(net)
            net.cuda()
        opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
        scheduler = StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)
        criterion = nn.MSELoss()

        logger.info('epoch   lr    | train_l  test_l |    acc     r2')
        res = []
        for epoch in range(EPOCHS):
            starttime = time.time()
            # train
            torch.set_grad_enabled(True)
            net.train()
            for x_self_body_batch, x_self_face_batch, x_interact_body_batch, x_interact_face_batch, y_batch in train_loader:
                if torch.cuda.is_available():
                    x_self_body_batch = x_self_body_batch.cuda()
                    x_self_face_batch = x_self_face_batch.cuda()
                    x_interact_body_batch = x_interact_body_batch.cuda()
                    x_interact_face_batch = x_interact_face_batch.cuda()
                    y_batch = y_batch.clone().detach().to(torch.float).view(-1,1).cuda()

                y_out = net(x_self_body_batch, x_self_face_batch, x_interact_body_batch, x_interact_face_batch)
                train_loss = criterion(y_out, y_batch)

                net.zero_grad()
                train_loss.backward()
                opt.step()

            # test
            torch.set_grad_enabled(False)
            net.eval()
            total_loss = []
            true_label_list = []
            pred_label_list = []

            for x_self_body_batch, x_self_face_batch, x_interact_body_batch, x_interact_face_batch, y_batch in test_loader:
                if torch.cuda.is_available():
                    x_self_body_batch = x_self_body_batch.cuda()
                    x_self_face_batch = x_self_face_batch.cuda()
                    x_interact_body_batch = x_interact_body_batch.cuda()
                    x_interact_face_batch = x_interact_face_batch.cuda()
                    y_batch = y_batch.clone().detach().to(torch.float).view(-1,1).cuda()

                y_out = net(x_self_body_batch, x_self_face_batch, x_interact_body_batch, x_interact_face_batch)
                test_loss = criterion(y_out, y_batch)
                total_loss.append(test_loss)

                true_label_list.append(y_batch.cpu().detach().numpy())
                pred_label_list.append(y_out.cpu().detach().numpy())

            mean_loss = sum(total_loss) / total_loss.__len__()
            y_true = np.concatenate(true_label_list)
            y_pred = np.concatenate(pred_label_list)
            acc, r2 = evaluate_res(y_true, y_pred, label_type)

            net.train()

            # logs
            logger.info(
                f'[{epoch+1:02d}/{EPOCHS:02d}] '
                f'{opt.param_groups[0]["lr"]:.0e} | '
                f'{train_loss.item(): 2.4f} {mean_loss.item(): 2.4f} | '
                f'{acc:.4f} {r2:.4f}')
            writer.add_scalars(f'{timestamp}/{fold}', {
                'train loss': train_loss.item(),
                'test loss': mean_loss.item()
            }, epoch)
            res.append([epoch, train_loss.item(), mean_loss.item(), acc, r2])
            res_overall[fold] = [acc, r2]

            scheduler.step()

            break

        writer.close()
        save_results(res, fold, output_path, label_type, trait)

        weight_path = f'{ckpt_save_path}{trait}_{args.model}_fold{fold}.pth'
        net.cpu()
        torch.save(
            {
                'epoch': EPOCHS,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': opt.state_dict()
            }, weight_path)
        
        break

    mean_acc, mean_r2 = 0.0, 0.0
    for _, value in res_overall.items():
        mean_acc += value[0]
        mean_r2 += value[1]
    mean_acc = mean_acc / fold_num
    mean_r2 = mean_r2 / fold_num
    logger.info(
        f'Average: acc={mean_acc:.4f} b_acc={mean_r2:.4f}\n'
    )


def train_cla(model, task, label_type, trait, timestamp, output_path):
    """Train model."""
    ckpt_save_path = f'{output_path}ckpts/'
    os.makedirs(ckpt_save_path, exist_ok=True)
    writer = SummaryWriter('runs')

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

        train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=NUM_WORKERS)
        test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE,
                                 shuffle=True, num_workers=NUM_WORKERS)

        net = model()
        if torch.cuda.is_available():
            net = nn.DataParallel(net)
            net.cuda()
        opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
        scheduler = StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)
        criterion = nn.CrossEntropyLoss()

        logger.info(
            'epoch   lr    | train_l  test_l |    acc  b_acc      p      r     f1    auc'
        )
        res = []
        for epoch in range(EPOCHS):
            starttime = time.time()
            # train
            torch.set_grad_enabled(True)
            net.train()
            for x_self_body_batch, x_self_face_batch, x_interact_body_batch, x_interact_face_batch, y_batch in train_loader:
                if torch.cuda.is_available():
                    x_self_body_batch = x_self_body_batch.cuda()
                    x_self_face_batch = x_self_face_batch.cuda()
                    x_interact_body_batch = x_interact_body_batch.cuda()
                    x_interact_face_batch = x_interact_face_batch.cuda()
                    y_batch = y_batch.clone().detach().to(torch.int64).cuda()

                y_out = net(x_self_body_batch, x_self_face_batch, x_interact_body_batch, x_interact_face_batch)
                train_loss = criterion(y_out, y_batch)

                net.zero_grad()
                train_loss.backward()
                opt.step()

            # test
            torch.set_grad_enabled(False)
            net.eval()
            total_loss = []
            true_label_list = []
            pred_label_list = []

            for x_self_body_batch, x_self_face_batch, x_interact_body_batch, x_interact_face_batch, y_batch in test_loader:
                if torch.cuda.is_available():
                    x_self_body_batch = x_self_body_batch.cuda()
                    x_self_face_batch = x_self_face_batch.cuda()
                    x_interact_body_batch = x_interact_body_batch.cuda()
                    x_interact_face_batch = x_interact_face_batch.cuda()
                    y_batch = y_batch.clone().detach().to(torch.int64).cuda()

                y_out = net(x_self_body_batch, x_self_face_batch, x_interact_body_batch, x_interact_face_batch)
                test_loss = criterion(y_out, y_batch)
                total_loss.append(test_loss)

                predict = y_out.argmax(dim=1)
                true_label_list.append(y_batch.cpu().detach().numpy())
                pred_label_list.append(predict.cpu().detach().numpy())

            mean_loss = sum(total_loss) / total_loss.__len__()
            y_true = np.concatenate(true_label_list)
            y_pred = np.concatenate(pred_label_list)
            acc, bal_acc, p, r, f1, auc = evaluate_res(y_true, y_pred)

            net.train()

            # logs
            logger.info(
                f'[{epoch+1:02d}/{EPOCHS:02d}] '
                f'{opt.param_groups[0]["lr"]:.0e} | '
                f'{train_loss.item(): 2.4f} {mean_loss.item(): 2.4f} | '
                f'{acc:.4f} {bal_acc:.4f} {p:.4f} {r:.4f} {f1:.4f} {auc:.4f}')
            writer.add_scalars(f'{timestamp}/{fold}', {
                'train loss': train_loss.item(),
                'test loss': mean_loss.item()
            }, epoch)
            res.append([
                epoch,
                train_loss.item(),
                mean_loss.item(), acc, bal_acc, p, r, f1, auc
            ])
            res_overall[fold] = [acc, bal_acc, p, r, f1, auc]

            scheduler.step()

        writer.close()
        save_results(res, fold, output_path, trait)

        weight_path = f'{ckpt_save_path}{trait}_{args.model}_fold{fold}.pth'
        net.cpu()
        torch.save(
            {
                'epoch': EPOCHS,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': opt.state_dict()
            }, weight_path)

    mean_acc, mean_bal_acc, mean_p, mean_r, mean_f1, mean_auc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for _, value in res_overall.items():
        mean_acc += value[0]
        mean_bal_acc += value[1]
        mean_p += value[2]
        mean_r += value[3]
        mean_f1 += value[4]
        mean_auc += value[5]
    mean_acc = mean_acc / fold_num
    mean_bal_acc = mean_bal_acc / fold_num
    mean_p = mean_p / fold_num
    mean_r = mean_r / fold_num
    mean_f1 = mean_f1 / fold_num
    mean_auc = mean_auc / fold_num
    logger.info(
        f'Average: acc={mean_acc:.4f} b_acc={mean_bal_acc:.4f} p={mean_p:.4f} r={mean_r:.4f} f1={mean_f1:.4f} auc={mean_auc:.4f}\n'
    )


if __name__ == '__main__':
    # torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    args = make_parser().parse_args()
    timestamp = time.strftime('%y%m%d-%H%M%S', time.localtime())
    output_path = f'results/{timestamp}/'
    os.makedirs(output_path, exist_ok=True)
    logger.add(f'{output_path}/log.txt', format='{message}', level='INFO')

    # configure training
    models = {
        'SB1'                     : nets.SB1,
        'SB3'                     : nets.SB3,
        'SB_nl'                   : nets.SB_nl,
        'SBF_fc'                  : nets.SBF_fc,
        'SBF_nlfc'                : nets.SBF_nlfc,
        'SBF_nlfc_IBF_nlfc'       : nets.SBF_nlfc_IBF_nlfc,
        'SBF_nlfc_IBF_nlfc_d'     : nets.SBF_nlfc_IBF_nlfc_d,
        'SBF_nlfc_IBF_nlfc_d_reg' : nets.SBF_nlfc_IBF_nlfc_d_reg,
    }
    model = models[args.model]
    traits = ['O', 'C', 'E', 'A', 'N']

    # hyper-parameters
    EPOCHS = 10
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    STEP_SIZE = 10
    GAMMA = 0.1
    NUM_WORKERS = 4

    # call training
    if args.trait == 'ALL':
        for t in traits:
            logger.info(
                f'{"=========="*8}\n'
                f'Configuration:'
                f'\n  TASK          : {args.task}'
                f'\n  TRAIT         : {t}'
                f'\n  MODEL         : {args.model}'
                f'\n  BATCH SIZE    : {BATCH_SIZE}'
                f'\n  EPOCH         : {EPOCHS}'
                f'\n  LEARNING RATE : {LEARNING_RATE:.0e}, step={GAMMA}/{STEP_SIZE}\n'
            )
            if args.label_type == 'class':
                train_cla(model, args.task, args.label_type, t, timestamp, output_path)
            elif args.label_type == 'reg':
                train_reg(model, args.task, args.label_type, t, timestamp, output_path)
    else:
        logger.info(
            f'{"=========="*8}\n'
            f'Configuration:'
            f'\n  TASK          : {args.task}'
            f'\n  TRAIT         : {args.trait}'
            f'\n  MODEL         : {args.model}'
            f'\n  BATCH SIZE    : {BATCH_SIZE}'
            f'\n  EPOCH         : {EPOCHS}'
            f'\n  LEARNING RATE : {LEARNING_RATE:.0e}, step={GAMMA}/{STEP_SIZE}\n'
        )
        if args.label_type == 'class':
            train_cla(model, args.task, args.label_type, args.trait, timestamp, output_path)
        elif args.label_type == 'reg':
            train_reg(model, args.task, args.label_type, args.trait, timestamp, output_path)
