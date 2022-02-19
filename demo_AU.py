import os
import numpy as np
import os
import pandas as pd
def file_name_npy(file_dir):
    L=[]
    for dirpath, dirnames, filenames in os.walk(file_dir):
        for file in filenames:
            if os.path.splitext(file)[1] == '.avi':
                L.append(os.path.join(dirpath, file))
    # print(L)
    return L

if __name__ == '__main__':
    path = "/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/data_set_AU_UBFC_Phys_T1_T2/"
    print('path = ', path)
    dirs = os.listdir(path)  # /home/som/8T/DataSets/ubfc_phys/video/s16/
    # count = 0
    # data_set_bvp_T1 = []
    count = 0
    frame_num = 6090
    data_eyegaze_arr = []
    data_head_pos_arr = []
    data_au_arr = []
    for file in dirs:
        if file[-6:]=='T1.csv':
            print(file)
            data_path = path + file
            print(data_path)
            count = count+1
            data = pd.read_csv(data_path, header=1)
            print('data.shape = ', data.shape)
            data_eyegaze = data.values[0:frame_num, 5:7]
            data_head_pos = data.values[0:frame_num, 7:10]
            data_au = data.values[0:frame_num, 10:27]
            print('data_eyegaze.shape = ', data_eyegaze.shape)
            print('data_head_pos.shape = ', data_head_pos.shape)
            print('data_au.shape = ', data_au.shape)
            data_eyegaze_arr.append(data_eyegaze)
            data_head_pos_arr.append(data_head_pos)
            data_au_arr.append(data_au)
    data_eyegaze_arr = np.array(data_eyegaze_arr)
    data_head_pos_arr = np.array(data_head_pos_arr)
    data_au_arr = np.array(data_au_arr)
    print('data_eyegaze_arr.shape = ', data_eyegaze_arr.shape)
    print('data_head_pos_arr.shape = ', data_head_pos_arr.shape)
    print('data_au_arr.shape = ', data_au_arr.shape)
