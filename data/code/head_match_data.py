# upload to google drive
# /open_pose
import re
import numpy as np
import h5py
import os

# return file name, ex: framexx_heads


def fileName(num):
    num1, num2 = num.split('.')
    return num1

# return a row number


def frameNum(frame_path):
    sub = 'frame(.*)_head.jpg'
    a = re.findall(sub, frame_path)
    result = int(a[0])
    return result


datasets_path = '/content/drive/My Drive/open_pose/data'
dataset = os.listdir(datasets_path)
num = len(dataset)
print(num)
for each_dataset in dataset:
    print(each_dataset)
    dataset_path = datasets_path + '/' + each_dataset
    txt_file = dataset_path + '/xyz.txt'
    # read xyz.txt
    f = np.loadtxt(txt_file)
    #/heads
    head_dir = dataset_path + '/heads'
    heads = os.listdir(head_dir)
    for head in heads:
        cur_path = head_dir + '/' + head
        img = fileName(head)
        create_txt = head_dir + '/' + img + '.txt'
        # row number
        num = frameNum(head)
        roll = f[num][1]
        pitch = f[num][2]
        yaw = f[num][3]
        np.savetxt(create_txt, (roll, pitch, yaw))

    # print(each_dataset)
    print('---------- finish -----------')
