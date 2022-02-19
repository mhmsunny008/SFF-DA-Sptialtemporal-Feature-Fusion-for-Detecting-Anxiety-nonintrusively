import os
import numpy as np
import os
def file_name_npy(file_dir):
    L=[]
    for dirpath, dirnames, filenames in os.walk(file_dir):
        for file in filenames:
            if os.path.splitext(file)[1] == '.avi':
                L.append(os.path.join(dirpath, file))
    # print(L)
    return L

if __name__ == '__main__':
    path = "/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/data_UBFC_Phys_bvp/"
    print('path = ', path)
    dirs = os.listdir(path)  # /home/som/8T/DataSets/ubfc_phys/video/s16/
    count = 0
    data_set_bvp_T1 = []
    count = 0
    for file in dirs:
        # print(file)
        dirs_save = path + file + '/'
        dirs_sub = os.listdir(dirs_save)
        # print(dirs_sub)
        for sub_file in dirs_sub:
            if sub_file[-6:]=='T2.csv':
                print(sub_file[-6:])
                # print(sub_file)
                sub_path_sub_file_save = dirs_save + sub_file
                print(sub_path_sub_file_save)
                count =count+1
                # data_bvp_t1 =  np.loadtxt(sub_path_sub_file_save)
                # print('data_bvp_t1.shape = ', data_bvp_t1.shape)
                # data_set_bvp_T1.append(data_bvp_t1)
    # data_set_bvp_T1 = np.array(data_set_bvp_T1)
    # print('data_set_bvp_T1.shape = ', data_set_bvp_T1.shape)
    print(count)
    # x = []
    # y= data
    # for i in range(len(data)):
    #     x.append(i)
    # x = np.array(x)
    # import matplotlib.pyplot as plt
    # plt.plot(x, y)
    # # 第3步：显示图形
    # plt.show()