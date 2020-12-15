# upload to google drive
# /open_pose
import csv
import re
import numpy as np
import h5py
import os


def findCSV(datasets_path):
    #datasets_path = '/Users/shanfandi/Desktop/2019-10-15-14-20-47.bag'
    sub = '.*orientation.csv'
    datasets = os.listdir(datasets_path)
    for dataset in datasets:
        result = re.match(sub, dataset)
        if result != None:
            path = result.group()
            break
    return path


def num2num(num):
    num1, num2 = num.split('.')
    num = num1 + '.' + num2[0:1]
    return num


def frameNum(frame_path):
    sub = 'frame(.*)_head.jpg'
    result = re.findall(sub, frame_path)
    return result


datasets_path = '/content/drive/My Drive/open_pose/data'
dataset = os.listdir(datasets_path)
num = len(dataset)
print(num)
for each_dataset in dataset:
    dataset_path = datasets_path + '/' + each_dataset
    txt_file = dataset_path + '/xyz.txt'
    data_csv = dataset_path + '/img_raw_03.csv'
    a = findCSV(dataset_path)
    data3_csv = dataset_path + '/' + a
    with open(data_csv, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]

    data = np.array(rows)
    with open(data3_csv, 'r') as csvfile3:
        reader3 = csv.reader(csvfile3)
        rows3 = [row3 for row3 in reader3]

    a = []
    data3 = np.array(rows3)
    isMatch = False
# data -> img_raw_03.csv
    for i in range(0, len(data)):
        isMatch = False
        for j in range(0, len(data3)):
            time1 = format(float(data[i][1]), '.1f')
            time2 = format(float(data3[j][0]), '.1f')
            if time1 == time2:
                isMatch = True
            # b.append(i)
                frame_data = [str(time1), data3[j][1],
                              data3[j][2], data3[j][3]]
                a.append(frame_data)
                frame_data = []
                break
        if isMatch == False:
            time1 = format(float(data[i][1]), '.1f')
            # set yaw, pitch, roll to 0
            frame_data = [str(time1), '0', '0', '0']
            a.append(frame_data)
            frame_data = []
# a = np.array(a)
    np.savetxt(txt_file, a, fmt='%s')
    print(txt_file)
    print('---------- finish -----------')
