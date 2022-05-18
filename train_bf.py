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
from utils.prepare_data_bf import create_mhhri


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', dest='task', type=str)
    parser.add_argument('--model', dest='model', type=str)
    parser.add_argument('--trait', dest='trait', type=str)

    return parser


def save_results(res, fold, output_path, trait):
    csv_path = f'{output_path}/csv/'
    fig_path = f'{output_path}/figs/'
    os.makedirs(csv_path, exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)

    clms = [
        'epoch', 'train_loss', 'test_loss', 'acc', 'bal_acc', 'precision',
        'recall', 'f1', 'auc'
    ]
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


def evaluate_res(y_true, y_pred):
    acc = metrics.accuracy_score(y_true, y_pred)
    bal_acc = metrics.balanced_accuracy_score(y_true, y_pred)
    p, r, f1 = metrics.precision_recall_fscore_support(y_true, y_pred, average='macro')[:-1]
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)

    return acc, bal_acc, p, r, f1, auc


def train(model, trait, timestamp, output_path):
    """Train model."""
    ckpt_save_path = f'{output_path}ckpts/'
    os.makedirs(ckpt_save_path, exist_ok=True)
    writer = SummaryWriter('runs')

    fold_num = 6
    body_data_list = np.load(
        'data/data_list/session_user_level_body_list_2.npy', allow_pickle=True)
    face_data_list = np.load(
        'data/data_list/session_user_level_face_list_2.npy', allow_pickle=True)

    res_overall = {}
    for fold in range(fold_num):
        body_test_files = body_data_list[fold]
        body_train_list = np.delete(body_data_list, fold, axis=0)
        body_train_files = [item for row in body_train_list for item in row]

        face_test_files = face_data_list[fold]
        face_train_list = np.delete(face_data_list, fold, axis=0)
        face_train_files = [item for row in face_train_list for item in row]
        
        logger.info(
            f'fold {fold} - train_num={len(body_train_files)}, test_num={len(body_test_files)}'
        )

        train_data, test_data = create_mhhri(body_train_files, body_test_files,
                                             face_train_files, face_test_files,
                                             trait)

        train_loader = DataLoader(dataset=train_data,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
        test_loader = DataLoader(dataset=test_data,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)

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
            for x_body_batch, x_face_batch, y_batch in train_loader:
                if torch.cuda.is_available():
                    x_body_batch = x_body_batch.cuda()
                    x_face_batch = x_face_batch.cuda()
                    y_batch = y_batch.clone().detach().to(torch.int64).cuda()

                y_out = net(x_body_batch, x_face_batch)
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

            for x_body_batch, x_face_batch, y_batch in test_loader:
                if torch.cuda.is_available():
                    x_body_batch = x_body_batch.cuda()
                    x_face_batch = x_face_batch.cuda()
                    y_batch = y_batch.clone().detach().to(torch.int64).cuda()

                y_out = net(x_body_batch, x_face_batch)
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
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    args = make_parser().parse_args()
    timestamp = time.strftime('%y%m%d-%H%M%S', time.localtime())
    output_path = f'results/{timestamp}/'
    os.makedirs(output_path, exist_ok=True)
    logger.add(f'{output_path}/log.txt', format='{message}', level='INFO')

    # tasks
    tasks = ['acq', 'self']
    assert (args.task in tasks)

    # model
    models = {
        'BODY_NL_FACE_FC1': nets.BODY_NL_FACE_FC1,
        'BODY_FACE_FC1': nets.BODY_FACE_FC1
    }
    assert (args.model in list(models.keys()))
    model = models[args.model]

    # trait
    traits = ['O', 'C', 'E', 'A', 'N', 'ALL']
    assert (args.trait in traits)

    # hyper-parameters
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    STEP_SIZE = 10
    GAMMA = 0.1

    if args.trait == traits[-1]:
        for trait in traits[:-1]:
            print(trait)
            logger.info(
                f'{"=========="*8}\n'
                f'Configuration:'
                f'\n  TASK          : {args.task}'
                f'\n  TRAIT         : {args.trait} - {trait}'
                f'\n  MODEL         : {args.model}'
                f'\n  BATCH SIZE    : {BATCH_SIZE}'
                f'\n  EPOCH         : {EPOCHS}'
                f'\n  LEARNING RATE : {LEARNING_RATE:.0e}, step={GAMMA}/{STEP_SIZE}\n'
            )
            train(model, trait, timestamp, output_path)

    else:
        print(args.trait)
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
        train(model, args.trait, timestamp, output_path)
