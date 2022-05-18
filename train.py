import argparse
import time

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
from utils.prepare_data import create_mhhri


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', type=str)
    parser.add_argument('--trait', dest='trait', type=str)

    return parser


def save_results(res, fold):
    res = pd.DataFrame(res, columns=['epoch', 'train_loss', 'test_loss', 'acc', 'precision', 'recall', 'f1', 'auc'])

    res_save_path = f'figs/{timestamp}_{fold}.csv'
    res.to_csv(res_save_path, index=False)
    logger.info(f'Save results to {res_save_path}')

    fig, ax = plt.subplots()
    ax.plot(res['epoch'], res['train_loss'], label='train loss')
    ax.plot(res['epoch'], res['test_loss'], label='test loss')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.grid()
    ax.legend()
    fig.savefig(f'figs/{timestamp}_{fold}.png')

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
    fig.savefig(f'figs/{timestamp}_{fold}_smooth.png')


def evaluate_res(y_true, y_pred):
    acc = metrics.accuracy_score(y_true, y_pred)
    p, r, f1 = metrics.precision_recall_fscore_support(y_true, y_pred, average='macro')[:-1]
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)

    return acc, p, r, f1, auc


def train(model, timestamp):
    """Train model."""
    writer = SummaryWriter('runs')

    fold_num = 6
    data_list = np.load('data/data_list/session_user_level_list.npy', allow_pickle=True)

    res_overall = {}
    for fold in range(fold_num):
        test_files = data_list[fold]
        train_list = np.delete(data_list, fold, axis=0)
        train_files = [item for row in train_list for item in row]
        logger.info(f'fold {fold}: train_num={len(train_files)}, test_num={len(test_files)}')

        train_data, test_data = create_mhhri(train_files, test_files, trait)

        train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

        net = model()
        if torch.cuda.is_available():
            net = nn.DataParallel(net)
            net.cuda()
        opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
        scheduler = StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)
        criterion = nn.CrossEntropyLoss()

        res = []
        for epoch in range(EPOCHS):
            starttime = time.time()
            # train
            torch.set_grad_enabled(True)
            net.train()
            for x_batch, y_batch in train_loader:
                if torch.cuda.is_available():
                    x_batch = x_batch.cuda()
                    y_batch = y_batch.clone().detach().to(torch.int64).cuda()

                y_out = net(x_batch)
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

            for x_batch, y_batch in test_loader:
                if torch.cuda.is_available():
                    x_batch = x_batch.cuda()
                    y_batch = y_batch.clone().detach().to(torch.int64).cuda()

                y_out = net(x_batch)
                test_loss = criterion(y_out, y_batch)
                total_loss.append(test_loss)

                predict = y_out.argmax(dim=1)
                true_label_list.append(y_batch.cpu().detach().numpy())
                pred_label_list.append(predict.cpu().detach().numpy())

            mean_loss = sum(total_loss) / total_loss.__len__()
            y_true = np.concatenate(true_label_list)
            y_pred = np.concatenate(pred_label_list)
            acc, p, r, f1, auc = evaluate_res(y_true, y_pred)

            net.train()

            # logs
            logger.info(f'[{epoch+1:03d}/{EPOCHS:03d}] {time.time() - starttime:.2f}s '
                        f'lr={opt.param_groups[0]["lr"]:.0e}, '
                        f'train_loss={train_loss.item():.4f}, test_loss={mean_loss.item():.4f}, '
                        f'acc={acc:.4f}, p={p:.4f}, r={r:.4f}, f1={f1:.4f}, auc={auc:.4f}')
            writer.add_scalars(f'{timestamp}/{fold}',
                               {'train loss': train_loss.item(),
                                'test loss': mean_loss.item()}, epoch)
            res.append([epoch, train_loss.item(), mean_loss.item(), acc, p, r, f1, auc])
            res_overall[fold] = [acc, p, r, f1, auc]

            scheduler.step()

        writer.close()
        save_results(res, fold)

        weight_path = f'checkpoints/{args.trait}_{args.model}_fold{fold}_{timestamp}.pth'
        logger.info(f'Save checkpoints to {weight_path}')
        net.cpu()
        torch.save({'epoch': EPOCHS,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': opt.state_dict()
                    }, weight_path)
        break

    mean_acc, mean_p, mean_r, mean_f1, mean_auc = 0.0, 0.0, 0.0, 0.0, 0.0
    for key, value in res_overall.items():
        logger.info(f'Fold {key}: {value}')
        mean_acc += value[0]
        mean_p += value[1]
        mean_r += value[2]
        mean_f1 += value[3]
        mean_auc += value[4]
    mean_acc = mean_acc / fold_num
    mean_p = mean_p / fold_num
    mean_r = mean_r / fold_num
    mean_f1 = mean_f1 / fold_num
    mean_auc = mean_auc / fold_num
    logger.info(f'Average: mean_acc={mean_acc:.4f}, mean_p={mean_p:.4f}, mean_r={mean_r:.4f},mean_f1={mean_f1:.4f}, mean_auc={mean_auc:.4f}')


if __name__ == '__main__':
    # torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    args = make_parser().parse_args()
    timestamp = time.strftime('%y%m%d-%H%M%S', time.localtime())
    logger.add(f'logs/{timestamp}.txt', format='{message}', level='INFO')

    # trait
    trait = args.trait
    traits = ['E', 'A', 'C', 'N', 'O']
    assert(trait in traits)

    # model
    models = {'NL_FC1': nets.NL_FC1, 'FC1': nets.FC1, 'FC3': nets.FC3}
    assert(args.model in list(models.keys()))
    model = models[args.model]

    # hyper-parameters
    EPOCHS = 20
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    STEP_SIZE = 10
    GAMMA = 0.1

    logger.info(f'Configuration:'
                f'\n  TRAIT         : {trait}'
                f'\n  MODEL         : {model.__name__}'
                f'\n  BATCH SIZE    : {BATCH_SIZE}'
                f'\n  EPOCH         : {EPOCHS}'
                f'\n  LEARNING RATE : {LEARNING_RATE:.0e}, reduce by {GAMMA} every {STEP_SIZE} steps')

    train(model, timestamp)
