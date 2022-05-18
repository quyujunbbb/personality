import os

from natsort import natsorted
import numpy as np
import pandas as pd


def eval_epoch():
    runs = ['220505-215605', '220506-133429', '220510-165009', '220510-220020',
            '220511-015005', '220511-055505', '220511-133716', '220511-180027',
            '220512-144010']
    cols = ['epoch',
            'train_loss','test_loss',
            'acc','bal_acc','precision','recall','f1','auc']

    overall_res = pd.DataFrame(columns=cols)
    for run in runs:
        run_path = f'results/{run}/csv/'

        O_fold0 = pd.read_csv(run_path+'O_fold0.csv')
        O_fold1 = pd.read_csv(run_path+'O_fold1.csv')
        O_fold2 = pd.read_csv(run_path+'O_fold2.csv')
        O_fold3 = pd.read_csv(run_path+'O_fold3.csv')
        O_fold4 = pd.read_csv(run_path+'O_fold4.csv')
        O_fold5 = pd.read_csv(run_path+'O_fold5.csv')

        C_fold0 = pd.read_csv(run_path+'C_fold0.csv')
        C_fold1 = pd.read_csv(run_path+'C_fold1.csv')
        C_fold2 = pd.read_csv(run_path+'C_fold2.csv')
        C_fold3 = pd.read_csv(run_path+'C_fold3.csv')
        C_fold4 = pd.read_csv(run_path+'C_fold4.csv')
        C_fold5 = pd.read_csv(run_path+'C_fold5.csv')

        E_fold0 = pd.read_csv(run_path+'E_fold0.csv')
        E_fold1 = pd.read_csv(run_path+'E_fold1.csv')
        E_fold2 = pd.read_csv(run_path+'E_fold2.csv')
        E_fold3 = pd.read_csv(run_path+'E_fold3.csv')
        E_fold4 = pd.read_csv(run_path+'E_fold4.csv')
        E_fold5 = pd.read_csv(run_path+'E_fold5.csv')

        A_fold0 = pd.read_csv(run_path+'A_fold0.csv')
        A_fold1 = pd.read_csv(run_path+'A_fold1.csv')
        A_fold2 = pd.read_csv(run_path+'A_fold2.csv')
        A_fold3 = pd.read_csv(run_path+'A_fold3.csv')
        A_fold4 = pd.read_csv(run_path+'A_fold4.csv')
        A_fold5 = pd.read_csv(run_path+'A_fold5.csv')

        N_fold0 = pd.read_csv(run_path+'N_fold0.csv')
        N_fold1 = pd.read_csv(run_path+'N_fold1.csv')
        N_fold2 = pd.read_csv(run_path+'N_fold2.csv')
        N_fold3 = pd.read_csv(run_path+'N_fold3.csv')
        N_fold4 = pd.read_csv(run_path+'N_fold4.csv')
        N_fold5 = pd.read_csv(run_path+'N_fold5.csv')

        O_avg = pd.DataFrame(columns=cols)
        C_avg = pd.DataFrame(columns=cols)
        E_avg = pd.DataFrame(columns=cols)
        A_avg = pd.DataFrame(columns=cols)
        N_avg = pd.DataFrame(columns=cols)

        for ep in range(10):
            O_temp = (O_fold0.iloc[ep,:] + O_fold1.iloc[ep,:] + O_fold2.iloc[ep,:] + O_fold3.iloc[ep,:] + O_fold4.iloc[ep,:] + O_fold5.iloc[ep,:]) / 6
            O_avg = O_avg.append(O_temp)

            C_temp = (C_fold0.iloc[ep,:] + C_fold1.iloc[ep,:] + C_fold2.iloc[ep,:] + C_fold3.iloc[ep,:] + C_fold4.iloc[ep,:] + C_fold5.iloc[ep,:]) / 6
            C_avg = C_avg.append(C_temp)

            E_temp = (E_fold0.iloc[ep,:] + E_fold1.iloc[ep,:] + E_fold2.iloc[ep,:] + E_fold3.iloc[ep,:] + E_fold4.iloc[ep,:] + E_fold5.iloc[ep,:]) / 6
            E_avg = E_avg.append(E_temp)

            A_temp = (A_fold0.iloc[ep,:] + A_fold1.iloc[ep,:] + A_fold2.iloc[ep,:] + A_fold3.iloc[ep,:] + A_fold4.iloc[ep,:] + A_fold5.iloc[ep,:]) / 6
            A_avg = A_avg.append(A_temp)

            N_temp = (N_fold0.iloc[ep,:] + N_fold1.iloc[ep,:] + N_fold2.iloc[ep,:] + N_fold3.iloc[ep,:] + N_fold4.iloc[ep,:] + N_fold5.iloc[ep,:]) / 6
            N_avg = N_avg.append(N_temp)

        O_min_epoch = O_avg.idxmin(axis=0)
        O_max_epoch = O_avg.idxmax(axis=0)
        O_best = pd.Series(['best', O_min_epoch[1], O_min_epoch[2], O_max_epoch[3], O_max_epoch[4], O_max_epoch[5], O_max_epoch[6], O_max_epoch[7], O_max_epoch[8]], index=O_avg.columns)
        O_avg = O_avg.append(O_best, ignore_index=True)

        C_min_epoch = C_avg.idxmin(axis=0)
        C_max_epoch = C_avg.idxmax(axis=0)
        C_best = pd.Series(['best', C_min_epoch[1], C_min_epoch[2], C_max_epoch[3], C_max_epoch[4], C_max_epoch[5], C_max_epoch[6], C_max_epoch[7], C_max_epoch[8]], index=C_avg.columns)
        C_avg = C_avg.append(C_best, ignore_index=True)

        E_min_epoch = E_avg.idxmin(axis=0)
        E_max_epoch = E_avg.idxmax(axis=0)
        E_best = pd.Series(['best', E_min_epoch[1], E_min_epoch[2], E_max_epoch[3], E_max_epoch[4], E_max_epoch[5], E_max_epoch[6], E_max_epoch[7], E_max_epoch[8]], index=E_avg.columns)
        E_avg = E_avg.append(E_best, ignore_index=True)

        A_min_epoch = A_avg.idxmin(axis=0)
        A_max_epoch = A_avg.idxmax(axis=0)
        A_best = pd.Series(['best', A_min_epoch[1], A_min_epoch[2], A_max_epoch[3], A_max_epoch[4], A_max_epoch[5], A_max_epoch[6], A_max_epoch[7], A_max_epoch[8]], index=A_avg.columns)
        A_avg = A_avg.append(A_best, ignore_index=True)

        N_min_epoch = N_avg.idxmin(axis=0)
        N_max_epoch = N_avg.idxmax(axis=0)
        N_best = pd.Series(['best', N_min_epoch[1], N_min_epoch[2], N_max_epoch[3], N_max_epoch[4], N_max_epoch[5], N_max_epoch[6], N_max_epoch[7], N_max_epoch[8]], index=N_avg.columns)
        N_avg = N_avg.append(N_best, ignore_index=True)

        avg = pd.DataFrame(columns=cols)

        for ep in range(10):
            avg_temp = (O_avg.iloc[ep,:] + C_avg.iloc[ep,:] + E_avg.iloc[ep,:] + A_avg.iloc[ep,:] + N_avg.iloc[ep,:]) / 5
            avg = avg.append(avg_temp, ignore_index=True)

        min_epoch = avg.idxmin(axis=0)
        max_epoch = avg.idxmax(axis=0)
        best = pd.Series(['best', min_epoch[1], min_epoch[2], max_epoch[3], max_epoch[4], max_epoch[5], max_epoch[6], max_epoch[7], max_epoch[8]], index=avg.columns)
        avg = avg.append(best, ignore_index=True)

        O_avg.to_csv(f'results/eval_epoch/{run}_O.csv', na_rep='NaN', index=False)
        C_avg.to_csv(f'results/eval_epoch/{run}_C.csv', na_rep='NaN', index=False)
        E_avg.to_csv(f'results/eval_epoch/{run}_E.csv', na_rep='NaN', index=False)
        A_avg.to_csv(f'results/eval_epoch/{run}_A.csv', na_rep='NaN', index=False)
        N_avg.to_csv(f'results/eval_epoch/{run}_N.csv', na_rep='NaN', index=False)
        avg.to_csv(f'results/eval_epoch/{run}_avg.csv', na_rep='NaN', index=False)

        overall_res = overall_res.append(best, ignore_index=True)

        # print(O_avg)
        # print(C_avg)
        # print(E_avg)
        # print(A_avg)
        # print(N_avg)
        # print(avg)

    print(overall_res)


if __name__ == '__main__':
    eval_epoch()
