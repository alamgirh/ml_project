import numpy as np
import scipy.io as sio
import os

# return file name, ex: framexx_heads


def fileName(num):
    num1, num2 = num.split('.')
    return num1


datasets_path = '/content/drive/My Drive/open_pose/data_with_txt_train'
dataset = os.listdir(datasets_path)
num = len(dataset)
print(num)
for each_dataset in dataset:
    print(each_dataset)
    dataset_path = datasets_path + '/' + each_dataset
    head_dir = dataset_path + '/heads'
    heads = os.listdir(head_dir)
    for head in heads:
        if 'txt' in head:
            txt = fileName(head)
            read_txt = head_dir + '/' + head
            save_path = head_dir + '/' + txt + '.mat'
            data = np.loadtxt(read_txt)
            roll = data[0]
            pitch = data[1]
            yaw = data[2]
            save_array = np.array([pitch, yaw, roll])
            sio.savemat(save_path, {'Pose_Para': save_array})
    print('---------- finish -----------')
