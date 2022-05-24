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

from models import nets_scratch
from utils.mhhri import split_filelists, create_mhhri


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', type=str)
    parser.add_argument('--task' , dest='task' , type=str)
    parser.add_argument('--trait', dest='trait', type=str)

    return parser


def save_results(res, fold, output_path, trait):
    os.makedirs(f'{output_path}/csv/', exist_ok=True)
    # os.makedirs(f'{output_path}/figs/', exist_ok=True)

    cols = [
        'epoch', 'train_loss', 'test_loss', 'acc_r', 'r2', 'acc_c', 'bal_acc',
        'precision', 'recall', 'f1', 'auc'
    ]

    res = pd.DataFrame(res, columns=cols)
    res.to_csv(f'{output_path}/csv/{trait}_fold{fold}.csv', index=False)

    # # draw loss
    # fig, ax = plt.subplots()
    # ax.plot(res['epoch'], res['train_loss'], label='train loss')
    # ax.plot(res['epoch'], res['test_loss'], label='test loss')
    # ax.set_xlabel('epochs')
    # ax.set_ylabel('loss')
    # ax.grid()
    # ax.legend()
    # fig.savefig(f'{output_path}/figs/{trait}_fold{fold}_loss.png')

    # # draw loss smooth
    # smooth_factor = 0.8
    # smooth_train = res['train_loss'].ewm(alpha=(1 - smooth_factor)).mean()
    # smooth_test = res['test_loss'].ewm(alpha=(1 - smooth_factor)).mean()

    # fig, ax = plt.subplots()
    # ax.plot(res['epoch'], smooth_train, label='train loss')
    # ax.plot(res['epoch'], smooth_test, label='test loss')
    # ax.set_xlabel('epochs')
    # ax.set_ylabel('loss')
    # ax.grid()
    # ax.legend()
    # fig.savefig(f'{output_path}/figs/{trait}_fold{fold}_loss_s.png')

    # # draw accuracy
    # fig, ax = plt.subplots()
    # ax.plot(res['epoch'], res['acc'], label='test accuray')
    # ax.set_xlabel('epochs')
    # ax.set_ylabel('test accuray')
    # ax.grid()
    # ax.legend()
    # fig.savefig(f'{output_path}/figs/{trait}_fold{fold}_acc.png')


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


def train(model, task, trait, timestamp, output_path):
    # ckpt_save_path = f'{output_path}ckpts/'
    # os.makedirs(ckpt_save_path, exist_ok=True)
    # writer = SummaryWriter('runs')

    fold_num = 6
    self_body_data_list = np.load('data/data_list/acq_self_body.npy', allow_pickle=True)
    self_face_data_list = np.load('data/data_list/acq_self_face.npy', allow_pickle=True)
    interact_body_data_list = np.load('data/data_list/acq_interact_body.npy', allow_pickle=True)
    interact_face_data_list = np.load('data/data_list/acq_interact_face.npy', allow_pickle=True)

    logger.info(
        f'fold    time   | train_l  test_l |    acc     r2 |    '
        f'acc  b_acc      p      r     f1    auc'
    )

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

        r3d = nets_scratch.ResNet3D(num_classes=400)
        r3d.load_state_dict(torch.load('pretrained/i3d_r50_kinetics.pth'))

        net = model(r3d)
        if torch.cuda.is_available():
            net = nn.DataParallel(net)
            net.cuda()
        opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
        scheduler = StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)
        criterion = nn.MSELoss()

        res = []
        for epoch in range(EPOCHS):
            starttime = time.time()
            # train
            torch.set_grad_enabled(True)
            net.train()
            for s_body_batch, _, i_body_batch, _, y_batch in train_loader:
                if torch.cuda.is_available():
                    s_body_batch = s_body_batch.permute(0, 4, 1, 2, 3)
                    i_body_batch = i_body_batch.permute(0, 4, 1, 2, 3)
                    s_body_batch = s_body_batch.cuda()
                    i_body_batch = i_body_batch.cuda()
                    y_batch = y_batch.clone().detach().to(torch.float).view(-1, 1).cuda()

                y_out = net(s_body_batch, i_body_batch)
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

            for s_body_batch, _, i_body_batch, _, y_batch in test_loader:
                if torch.cuda.is_available():
                    s_body_batch = s_body_batch.permute(0, 4, 1, 2, 3)
                    i_body_batch = i_body_batch.permute(0, 4, 1, 2, 3)
                    s_body_batch = s_body_batch.cuda()
                    i_body_batch = i_body_batch.cuda()
                    y_batch = y_batch.clone().detach().to(torch.float).view(-1, 1).cuda()

                y_out = net(s_body_batch, i_body_batch)
                test_loss = criterion(y_out, y_batch)
                total_loss.append(test_loss)

                true_label_list.append(y_batch.cpu().detach().numpy())
                pred_label_list.append(y_out.cpu().detach().numpy())

            mean_loss = sum(total_loss) / total_loss.__len__()
            y_true = np.concatenate(true_label_list)
            y_pred = np.concatenate(pred_label_list)
            acc_r, r2, acc_c, bal_acc, p, r, f1, auc = evaluate_res(
                y_true, y_pred, trait, fold, output_path)

            net.train()

            # logs
            logger.info(
                f'fold {fold} {time.time() - starttime:.1f}s | '
                f'{train_loss.item(): 2.4f} {mean_loss.item(): 2.4f} | '
                f'{acc_r:.4f} {r2:.4f} | '
                f'{acc_c:.4f} {bal_acc:.4f} {p:.4f} {r:.4f} {f1:.4f} {auc:.4f}'
            )
            # writer.add_scalars(f'{timestamp}/{fold}', {
            #     'train loss': train_loss.item(),
            #     'test loss': mean_loss.item()
            # }, epoch)
            res.append([
                epoch,
                train_loss.item(),
                mean_loss.item(), acc_r, r2, acc_c, bal_acc, p, r, f1, auc
            ])
            res_overall[fold] = [acc_r, r2, acc_c, bal_acc, p, r, f1, auc]

            scheduler.step()

        # writer.close()
        save_results(res, fold, output_path, trait)

        # weight_path = f'{ckpt_save_path}{trait}_{args.model}_fold{fold}.pth'
        # net.cpu()
        # torch.save(
        #     {
        #         'epoch': EPOCHS,
        #         'model_state_dict': net.state_dict(),
        #         'optimizer_state_dict': opt.state_dict()
        #     }, weight_path)

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

    args = make_parser().parse_args()
    timestamp = time.strftime('%y%m%d-%H%M%S', time.localtime())
    output_path = f'results/{timestamp}/'
    os.makedirs(output_path, exist_ok=True)
    logger.add(f'{output_path}/log.txt', format='{message}', level='INFO')

    # configure training
    models = {
        'MyModelBody': nets_scratch.MyModelBody,
    }
    model = models[args.model]
    traits = ['O', 'C', 'E', 'A', 'N']

    # hyper-parameters
    EPOCHS = 1
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    STEP_SIZE = 10
    GAMMA = 0.1
    NUM_WORKERS = 2

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
                f'\n  LEARNING RATE : {LEARNING_RATE:.0e}, step={GAMMA}/{STEP_SIZE}'
            )
            train(model, args.task, t, timestamp, output_path)
    else:
        logger.info(
            f'{"=========="*8}\n'
            f'Configuration:'
            f'\n  TASK          : {args.task}'
            f'\n  TRAIT         : {args.trait}'
            f'\n  MODEL         : {args.model}'
            f'\n  BATCH SIZE    : {BATCH_SIZE}'
            f'\n  EPOCH         : {EPOCHS}'
            f'\n  LEARNING RATE : {LEARNING_RATE:.0e}, step={GAMMA}/{STEP_SIZE}'
        )
        train(model, args.task, args.trait, timestamp, output_path)
