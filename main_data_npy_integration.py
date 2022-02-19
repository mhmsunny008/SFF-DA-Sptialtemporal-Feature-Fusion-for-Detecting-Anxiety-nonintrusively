import os
import numpy as np
def file_name_npy(file_dir):
    L=[]
    for dirpath, dirnames, filenames in os.walk(file_dir):
        for file in filenames:
            if os.path.splitext(file)[1] == '.npy':
                L.append(os.path.join(dirpath, file))
    # print(L)
    return L

if __name__ == '__main__':
    path = "/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/data_set_UBFC_Phys/"
    # print('path = ', path)
    dirs = os.listdir(path)  # /home/som/8T/DataSets/ubfc_phys/video/s16/
    eyeLeft_ROI_arr = []
    eyeLeft_ROI_len_index = []
    ID_arr = []
    x_arr, y_arr, z_arr = [], [], []
    for file in dirs:

        if file[-2:]=='T1':
            # print(file[-2:])
            # print(file)
            dirs_save = path + file + '/'
            print(dirs_save)
            dirs_sub = os.listdir(dirs_save)
            for sub_file in dirs_sub:
                if sub_file=='loc':
                    sub_path_eyeLeft_ROI_save = dirs_save + sub_file +'/'
                    print(sub_path_eyeLeft_ROI_save)
                    sub_path_eyeLeft_ROI = os.listdir(sub_path_eyeLeft_ROI_save)
                    for sub_sub_flie in sub_path_eyeLeft_ROI:
                        # eyeLeft_ROI, eyeRight, forehead, mouth, nose_ROI,
                        # print(sub_sub_flie)
                        # path_npy = sub_path_eyeLeft_ROI_save+sub_sub_flie
                        # file_npy = np.load(path_npy)
                        # print(file_npy.shape)
                        # eyeLeft_ROI_arr.append(file_npy[0:174 * 35, :, :, :])
                        # eyeLeft_ROI_len_index.append(file_npy.shape[0])
                        # file_arr = file[0:-3]
                        # print(file[0:-3])
                        # print(file_arr[5:len(file_arr)])
                        # ID_arr.append(file_arr[5:len(file_arr)])

                        # ippg
                        # path_npy = sub_path_eyeLeft_ROI_save+sub_sub_flie
                        # file_npy = np.load(path_npy)
                        # print(file_npy.shape)
                        # eyeLeft_ROI_arr.append(file_npy[:,0:174*35])
                        # eyeLeft_ROI_len_index.append(file_npy.shape[1])
                        # file_arr = file[0:-3]
                        # print(file[0:-3])
                        # print(file_arr[5:len(file_arr)])
                        # ID_arr.append(file_arr[5:len(file_arr)])

                        # loc
                        path_npy = sub_path_eyeLeft_ROI_save + sub_sub_flie
                        print(path_npy)
                        print(path_npy[-14:])
                        file_npy = np.load(path_npy)
                        print(file_npy.shape)
                        # eyeLeft_ROI_arr.append(file_npy[0:174 * 35, :])  # （6324, 468）
                        eyeLeft_ROI_len_index.append(file_npy.shape[0])

                        file_arr = file[0:-3]
                        print(file[0:-3])
                        print(file_arr[5:len(file_arr)])
                        ID_arr.append(file_arr[5:len(file_arr)])
                        if path_npy[-14:] == 'x_data_set.npy':
                            x_arr.append(file_npy[0:174 * 35, :])  # （6324, 468）
                        if path_npy[-14:] == 'y_data_set.npy':
                            y_arr.append(file_npy[0:174 * 35, :])  # （6324, 468）
                        if path_npy[-14:] == 'z_data_set.npy':
                            z_arr.append(file_npy[0:174 * 35, :])  # （6324, 468）


                # if sub_file == 'eyeLeft_ROI':  # T2:34   T1:40
                #     sub_path_eyeRight_ROI = dirs_save + sub_file + '/'

                # if sub_file == 'eyeRight_ROI':  # T2:34   T1:40
                #     sub_path_eyeRight_ROI = dirs_save + sub_file + '/'
                #
                # if sub_file == 'forehead_ROI':# T2:34   T1:40
                #     sub_path_forehead_ROI = dirs_save + sub_file + '/'

                # if sub_file == 'mouth_ROI':# T2:34   T1:40
                #     sub_path_mouth_ROI = dirs_save + sub_file + '/'

                # if sub_file == 'nose_ROI':# T2:34   T1:40
                #     sub_path_nose_ROI = dirs_save + sub_file + '/'

                # if sub_file == 'ippg_forehead':  # T2:34   T1:40
                #     sub_path_ippg_forehead = dirs_save + sub_file + '/'
                #
                # if sub_file == 'ippg_nose': # T2:34   T1:40
                #     sub_path_ippg_nose = dirs_save + sub_file + '/'
                #
                # if sub_file == 'loc':# T2:34   T1:40
                #     sub_path_loc = dirs_save + sub_file + '/'
                #

    # eyeLeft_ROI_arr = np.array(eyeLeft_ROI_arr) # 178 second * 35 FPS = 6230 frame
    # print('eyeLeft_ROI_arr.shape = ', eyeLeft_ROI_arr.shape)
    # eyeLeft_ROI_len_index = np.array(eyeLeft_ROI_len_index)
    # max_len = np.max(eyeLeft_ROI_len_index)
    # min_len = np.min(eyeLeft_ROI_len_index)
    # print('max_len = ', max_len)
    # print('min_len = ', min_len)
    # ID_arr = np.sort(ID_arr,axis=0) # eyeLeft_ROI_arr 缺少s34
    # print(ID_arr)
    # print(len(ID_arr))
    # save_path_img_ippg_nose = '/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/forehead_ROI_T1_56_6125_40_40pixels/'
    # np.save(save_path_img_ippg_nose +  'forehead_ROI' + '.npy',
    #         np.array(eyeLeft_ROI_arr))

    x_arr = np.array(x_arr) # 178 second * 35 FPS = 6230 frame
    y_arr = np.array(y_arr)  # 178 second * 35 FPS = 6230 frame
    z_arr = np.array(z_arr)  # 178 second * 35 FPS = 6230 frame
    print('x_arr.shape = ', x_arr.shape)
    print('y_arr.shape = ', y_arr.shape)
    print('z_arr.shape = ', z_arr.shape)

    eyeLeft_ROI_len_index = np.array(eyeLeft_ROI_len_index)
    max_len = np.max(eyeLeft_ROI_len_index)
    min_len = np.min(eyeLeft_ROI_len_index)
    print('max_len = ', max_len)
    print('min_len = ', min_len)
    ID_arr = np.sort(ID_arr,axis=0) # eyeLeft_ROI_arr 缺少s34
    print(ID_arr)
    print(len(ID_arr))
    save_path_img_ippg_nose = '/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/loc_T1_56_6090_468/'
    np.save(save_path_img_ippg_nose +  'loc_x_T1_56_6090_468' + '.npy',
            np.array(x_arr))
    np.save(save_path_img_ippg_nose + 'loc_y_T1_56_6090_468' + '.npy',
            np.array(y_arr))
    np.save(save_path_img_ippg_nose + 'loc_z_T1_56_6090_468' + '.npy',
            np.array(z_arr))




