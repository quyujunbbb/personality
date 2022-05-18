import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from natsort import natsorted
from PIL import Image
from torch import nn
from torch.autograd import Variable

# from models import nets
# from models.resnet import i3_res50



# img1 = np.load('features/visualize/res4_1.npy')
# img2 = np.load('features/visualize/res4_2.npy')
# img1 = torch.tensor(img1).unsqueeze(0)
# img2 = torch.tensor(img2).unsqueeze(0)

# pretrained_path = 'checkpoints/extraversion_NL_FC1_a_220413-145247.pth'
# net = nets.NL_FC1()
# net = nn.DataParallel(net)
# state_dict = torch.load(pretrained_path)
# net.load_state_dict(torch.load(pretrained_path)['model_state_dict'])
# net.cuda()
# net.train(False)

# out1 = net(img1).squeeze(0).cpu().detach().numpy()
# out2 = net(img2).squeeze(0).cpu().detach().numpy()
# print(np.shape(out1))
# print(np.shape(out2))
# np.save('nl_1', out1)
# np.save('nl_2', out2)


# path = 'features/visualize/'
# files = natsorted(os.listdir(path))

# for file in files:
#     map_name = file.split('.')[0]
#     feature_map = torch.tensor(np.load(path + file))
#     print(feature_map.shape)

#     feature_map = feature_map.squeeze(0)
#     gray_scale = torch.sum(feature_map,0)
#     gray_scale = gray_scale / feature_map.shape[0]
#     processed = gray_scale.data.cpu().numpy()

#     fig = plt.figure(figsize=(30, 50))
#     for i in range(len(processed)):
#         a = fig.add_subplot(5, 4, i+1)
#         imgplot = plt.imshow(processed[i])
#     plt.savefig(f'{map_name}.png', bbox_inches='tight')


def load_frame(frame_file):
    data = Image.open(frame_file)
    data = np.array(data)
    data = data.astype(float)
    data = (data * 2 / 255) - 1
    assert(data.max() <= 1.0 and data.min() >= -1.0)
    return data


def load_rgb_batch(frames_dir, rgb_files, frame_indices):
    batch_data = np.zeros(frame_indices.shape + (224, 224, 3))
    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):
            batch_data[i, j, :, :, :] = load_frame(
                os.path.join(frames_dir, rgb_files[frame_indices[i][j]]))
    return batch_data


def forward_batch(i3d, b_data):
    b_data = b_data.transpose([0, 4, 1, 2, 3])
    b_data = torch.from_numpy(b_data)  # [b,c,t,h,w]
    with torch.no_grad():
        b_data = Variable(b_data.cuda()).float()
        inp = {'frames': b_data}
        features = i3d(inp)
    return features.cpu().numpy()


def extract_feature_map():
    pretrained_path = 'pretrained/i3d_r50_kinetics.pth'
    sample_folders = 'data/visualize_samples/'

    i3d = i3_res50(400, pretrained_path)
    i3d.cuda()
    i3d.train(False)

    samples = natsorted(os.listdir(sample_folders))
    for sample in samples:
        image_path = f'{sample_folders}{sample}/'

        chunk_size = 32
        batch_size = 16

        images = natsorted([i for i in os.listdir(image_path)])
        frame_cnt = len(images)
        assert(frame_cnt >= chunk_size)

        clipped_length = ((frame_cnt - chunk_size) // chunk_size) * chunk_size
        frame_indices = []
        for i in range(clipped_length // chunk_size + 1):
            frame_indices.append([j for j in range(i * chunk_size, i * chunk_size + chunk_size)])
        frame_indices = np.array(frame_indices)
        chunk_num = frame_indices.shape[0]
        batch_num = int(np.ceil(chunk_num / batch_size))
        frame_indices = np.array_split(frame_indices, batch_num, axis=0)
        
        full_features = []
        for batch_id in range(batch_num):
            batch_data = load_rgb_batch(image_path, images, frame_indices[batch_id])
            temp = forward_batch(i3d, batch_data)
            full_features.append(temp)

        # [batch_num, batch_size, ...] --> [batch_num x batch_size, ...]
        full_features = np.concatenate(full_features, axis=0)
        
        for i, feature in enumerate(full_features):
            np.save(f'res1_{sample}', feature)


def visualize_acquitance_label():
    label_path = 'data/annotations/acquitance_label.csv'
    labels = pd.read_csv(label_path)
    print(labels.describe())

    print(labels['O'].value_counts())
    print(labels['C'].value_counts())
    print(labels['E'].value_counts())
    print(labels['A'].value_counts())
    print(labels['N'].value_counts())
    #     O   C   E   A   N
    # 0   9  11  10   9  10 
    # 1   9   7   8   9   8


def data_distribution_in_folds():
    label_path = 'data/annotations/acquitance_label_session.csv'
    labels = pd.read_csv(label_path)

    # folds = [['S01_kinect_1', 'S02_kinect_2', 'S11_kinect_2', 'S12_kinect_1'],
    #          ['S02_kinect_1', 'S04_kinect_1', 'S10_kinect_2', 'S07_kinect_2'],
    #          ['S01_kinect_2', 'S05_kinect_2', 'S07_kinect_1', 'S10_kinect_1'],
    #          ['S03_kinect_2', 'S11_kinect_1', 'S05_kinect_1', 'S08_kinect_1'],
    #          ['S03_kinect_1', 'S06_kinect_2', 'S09_kinect_2', 'S08_kinect_2'],
    #          ['S04_kinect_2', 'S06_kinect_1', 'S09_kinect_1', 'S12_kinect_2']]

    folds = [['S01_kinect_1', 'S02_kinect_2', 'S03_kinect_1', 'S06_kinect_1'],
             ['S01_kinect_2', 'S02_kinect_1', 'S11_kinect_2', 'S12_kinect_1'],
             ['S03_kinect_2', 'S05_kinect_1', 'S07_kinect_1', 'S12_kinect_2'],
             ['S04_kinect_1', 'S07_kinect_2', 'S09_kinect_1', 'S10_kinect_1'],
             ['S04_kinect_2', 'S05_kinect_2', 'S08_kinect_1', 'S10_kinect_2'],
             ['S06_kinect_2', 'S08_kinect_2', 'S09_kinect_2', 'S11_kinect_1']]

    for fold_idx, fold in enumerate(folds):
        print(f'fold {fold_idx+1}')
        test_list = fold
        # print(test_list)
        test_label = pd.DataFrame()
        for test_item in test_list:
            test_label = test_label.append(labels[labels['session'] == test_item], ignore_index=True)
        # print(test_label)
        
        train_list = np.delete(folds, fold_idx, axis=0)
        train_list = [item for row in train_list for item in row]
        # print(train_list)
        train_label = pd.DataFrame()
        for train_item in train_list:
            train_label = train_label.append(labels[labels['session'] == train_item], ignore_index=True)
        # print(train_label)

        print('train')
        count_O = np.array(np.unique(np.array(train_label['O']), return_counts=True)).astype(int)[1]
        count_C = np.array(np.unique(np.array(train_label['C']), return_counts=True)).astype(int)[1]
        count_E = np.array(np.unique(np.array(train_label['E']), return_counts=True)).astype(int)[1]
        count_A = np.array(np.unique(np.array(train_label['A']), return_counts=True)).astype(int)[1]
        count_N = np.array(np.unique(np.array(train_label['N']), return_counts=True)).astype(int)[1]

        print(count_O)
        print(count_C)
        print(count_E)
        print(count_A)
        print(count_N)

        print('test')
        count_O = np.array(np.unique(np.array(test_label['O']), return_counts=True)).astype(int)[1]
        count_C = np.array(np.unique(np.array(test_label['C']), return_counts=True)).astype(int)[1]
        count_E = np.array(np.unique(np.array(test_label['E']), return_counts=True)).astype(int)[1]
        count_A = np.array(np.unique(np.array(test_label['A']), return_counts=True)).astype(int)[1]
        count_N = np.array(np.unique(np.array(test_label['N']), return_counts=True)).astype(int)[1]

        print(count_O)
        print(count_C)
        print(count_E)
        print(count_A)
        print(count_N)

        print()

data_distribution_in_folds()
