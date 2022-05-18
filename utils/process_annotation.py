import os

import pandas as pd
from natsort import natsorted


def normalize_reg_labels(input_file):
    root_path = 'data/annotations/'
    task = input_file.split('.')[0].split('_')[0]
    labels = pd.read_csv(root_path + input_file)

    out_path = f'data/annotations/{task}_reg_session_norm.csv'

    columns = ['E', 'A', 'C', 'N', 'O']
    columns_full = ['session', 'id', 'E', 'A', 'C', 'N', 'O']
    labels_out_1 = pd.DataFrame(columns=columns_full)
    labels_out_2 = pd.DataFrame(columns=columns_full)
    for col in columns:
        labels_out_1[col] = (labels[col] - labels[col].mean()) / labels[col].std()
        labels_out_2[col] = (labels[col] - 1) / (10 - 1)

    labels_out_1['session'], labels_out_1['id'] = labels['session'], labels['id']
    labels_out_2['session'], labels_out_2['id'] = labels['session'], labels['id']

    print(labels_out_1)
    print(labels_out_2)
    labels_out_2.round(6).to_csv(out_path, index=False)


def generate_class_labels(input_file):
    root_path = 'data/annotations/'
    task = input_file.split('.')[0].split('_')[0]
    labels = pd.read_csv(root_path + input_file)

    out_path = f'data/annotations/{task}_class.csv'

    mean = labels.mean()
    labels.loc[labels.E  < mean.E, 'E'] = 0
    labels.loc[labels.E >= mean.E, 'E'] = 1
    labels.loc[labels.A  < mean.A, 'A'] = 0
    labels.loc[labels.A >= mean.A, 'A'] = 1
    labels.loc[labels.C  < mean.C, 'C'] = 0
    labels.loc[labels.C >= mean.C, 'C'] = 1
    labels.loc[labels.N  < mean.N, 'N'] = 0
    labels.loc[labels.N >= mean.N, 'N'] = 1
    labels.loc[labels.O  < mean.O, 'O'] = 0
    labels.loc[labels.O >= mean.O, 'O'] = 1

    print(labels)
    labels.to_csv(out_path, index=False)


def preprocess_self_annotations():
    self_file = 'data/annotations/self/personality_labels.csv'

    self_out_path = 'data/annotations/self_reg.csv'
    labels_out = pd.DataFrame(columns=['id', 'E', 'A', 'C', 'N', 'O'])

    labels_in = pd.read_csv(self_file, header=None)
    # print(labels_in)

    # calculate revised
    labels_in.iloc[:, 0] = 10 - labels_in.iloc[:, 0] + 1
    labels_in.iloc[:, 6] = 10 - labels_in.iloc[:, 6] + 1
    labels_in.iloc[:, 2] = 10 - labels_in.iloc[:, 2] + 1
    labels_in.iloc[:, 3] = 10 - labels_in.iloc[:, 3] + 1
    labels_in.iloc[:, 4] = 10 - labels_in.iloc[:, 4] + 1
    # print(labels_in)

    # calculate big-five
    for idx in range(len(labels_in)):
        extraversion = (labels_in.iloc[idx, 0] + labels_in.iloc[idx, 5])/2
        agreebleness = (labels_in.iloc[idx, 1] + labels_in.iloc[idx, 6])/2
        conscientiousness = (labels_in.iloc[idx, 2] + labels_in.iloc[idx, 7])/2
        neuroticism = (labels_in.iloc[idx, 3] + labels_in.iloc[idx, 8])/2
        openness = (labels_in.iloc[idx, 4] + labels_in.iloc[idx, 9])/2

        row = {'id': f'U{idx+1:03d}',
               'E':extraversion,
               'A':agreebleness,
               'C':conscientiousness,
               'N':neuroticism,
               'O':openness,}
        labels_out = labels_out.append(row, ignore_index=True).round(6)

    print(labels_out)
    labels_out.to_csv(self_out_path, index=False)


def preprocess_acquitance_annotations():
    acquitance_path = 'data/annotations/acquitance/'
    acquitance_files = natsorted(os.listdir(acquitance_path))

    acquitance_out_path = 'data/annotations/acq_reg.csv'
    labels_out = pd.DataFrame(columns=['id', 'E', 'A', 'C', 'N', 'O'])

    for acquitance_file in acquitance_files:
        id = acquitance_file.split('.')[0]
        labels_in = pd.read_csv(acquitance_path + acquitance_file, header=None)
        labels_in = labels_in.transpose()
        # print(labels_in)

        # calculate revised
        labels_in.iloc[:, 0] = 10 - labels_in.iloc[:, 0] + 1
        labels_in.iloc[:, 6] = 10 - labels_in.iloc[:, 6] + 1
        labels_in.iloc[:, 2] = 10 - labels_in.iloc[:, 2] + 1
        labels_in.iloc[:, 3] = 10 - labels_in.iloc[:, 3] + 1
        labels_in.iloc[:, 4] = 10 - labels_in.iloc[:, 4] + 1
        # print(labels_in)

        # calcualte average among peers
        labels_in = labels_in.mean()
        # print(labels_in)

        # calculate big-five
        extraversion = (labels_in[0] + labels_in[5])/2
        agreebleness = (labels_in[1] + labels_in[6])/2
        conscientiousness = (labels_in[2] + labels_in[7])/2
        neuroticism = (labels_in[3] + labels_in[8])/2
        openness = (labels_in[4] + labels_in[9])/2

        row = {'id':id,
               'E':extraversion,
               'A':agreebleness,
               'C':conscientiousness,
               'N':neuroticism,
               'O':openness,}
        labels_out = labels_out.append(row, ignore_index=True).round(6)

    print(labels_out)
    labels_out.to_csv(acquitance_out_path, index=False)


if __name__ == '__main__':
    # preprocess_acquitance_annotations()
    # generate_class_labels(input_file='acq_reg.csv')
    normalize_reg_labels(input_file='acq_reg_session.csv')

    # preprocess_self_annotations()
    # generate_class_labels(input_file='self_reg.csv')
    normalize_reg_labels(input_file='self_reg_session.csv')
