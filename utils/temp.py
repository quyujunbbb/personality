import os
import numpy as np
import pandas as pd
from sklearn import metrics
from natsort import natsorted


if __name__ == '__main__':
    files = natsorted(os.listdir('results/220521-155359/pred/'))

    res = []
    for file in files:
        # save csv results
        y = pd.read_csv(f'results/220521-155359/pred/{file}')

        # evaluate regression results
        acc_r = 1 - np.sum(np.abs(y['y_true'] - y['y_pred'])) / len(y['y_true'])
        r2 = 1 - np.sum((y['y_true'] - y['y_pred'])**2) / len(y['y_true'])

        # evaluate classification results
        y['y_true_c'] = np.where(y['y_true']>=0.5, 1, 0)
        y['y_pred_c'] = np.where(y['y_pred']>=0.5, 1, 0)

        acc_c = metrics.accuracy_score(y['y_true_c'], y['y_pred_c'])
        bal_acc = metrics.balanced_accuracy_score(y['y_true_c'], y['y_pred_c'])
        p, r, f1 = metrics.precision_recall_fscore_support(
            y['y_true_c'], y['y_pred_c'], average='macro')[:-1]
        fpr, tpr, _ = metrics.roc_curve(y['y_true_c'], y['y_pred_c'])
        auc = metrics.auc(fpr, tpr)

        res.append([file.split('.')[0], acc_r, r2, acc_c, bal_acc, p, r, f1, auc])

    cols = ['name', 'acc_r', 'r2', 'acc_c', 'bal_acc', 'p', 'r', 'f1', 'auc']
    res = pd.DataFrame(res, columns=cols).round(4)
    res.to_csv(f'results/220521-155359/res.csv', index=False)
