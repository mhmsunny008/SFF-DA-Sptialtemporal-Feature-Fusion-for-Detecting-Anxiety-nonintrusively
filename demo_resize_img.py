import numpy as np
import cv2
def resize_img(data):
    img_arr = []
    img_arr_one = []
    img_arr_two = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            print('data[i,j].shape = ', data[i,j].shape)
            img =  cv2.resize(data[i,j], (10, 10))
            img_one = cv2.resize(data[i, j], (20, 20))
            img_two = cv2.resize(data[i, j], (30, 30))
            print('img.shape=', img.shape)
            img_arr.append(img)
            img_arr_one.append(img_one)
            img_arr_two.append(img_two)
    # img_arr = np.array(img_arr)
    img_arr = np.reshape(img_arr,(data.shape[0],data.shape[1],10,10,3))
    img_arr_one = np.reshape(img_arr_one, (data.shape[0], data.shape[1], 20, 20, 3))
    img_arr_two = np.reshape(img_arr_two, (data.shape[0], data.shape[1], 30, 30, 3))
    print('img_arr.shape = ', img_arr.shape)
    print('img_arr_one.shape = ', img_arr_one.shape)
    print('img_arr_two.shape = ', img_arr_two.shape)
    return img_arr, img_arr_one,img_arr_two


if __name__ == "__main__":
    # mouth_ROI_T1_56_6090_40_40pixels = np.load(
    #     r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/mouth_ROI_T1_56_6090_40_40pixels/'
    #     + 'mouth_ROI_T1_56_6090_40_40pixels.npy')

    # eyeLeft_ROI_T1_56_6090_40_40pixels = np.load(
    #     r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/eyeLeft_ROI_T1_56_6090_40_40pixels/'
    #     + 'eyeLeft_ROI_T1_56_6090_40_40pixels.npy')

    eyeRight_ROI_T1_56_6090_40_40pixels = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/eyeRight_ROI_T1_56_6090_40_40pixels/'
        + 'eyeRight_ROI_T1_56_6090_40_40pixels.npy')


    # mouth_ROI_T2 = np.load(
    #     r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/mouth_ROI_T2_56_6090_40_40pixels/'
    #     + 'mouth_ROI_T2_56_6090_40_40pixels.npy')

    # eyeLeft_ROI_T2 = np.load(
    #     r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/eyeLeft_ROI_T2_56_6090_40_40pixels/'
    #     + 'eyeLeft_ROI_T2_56_6090_40_40pixels.npy')



    # eyeRight_ROI_T2_56_6090_40_40pixels = np.load(
    #     r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/eyeRight_ROI_T2_56_6090_40_40pixels/'
    #     + 'eyeRight_ROI_T2_56_6090_40_40pixels.npy')
    print('eyeRight_ROI_T1_56_6090_40_40pixels.shape = ', eyeRight_ROI_T1_56_6090_40_40pixels.shape)
    # print('data_set_mouthRIO_T2.shape = ', data_set_mouthRIO_T2.shape)


    data, data_one, data_two = resize_img(eyeRight_ROI_T1_56_6090_40_40pixels)

    save_path ='/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/eyeRight_ROI_T1_56_6090_40_40pixels/'

    np.save(save_path + 'eyeRight_ROI_T1_56_6090_10_10pixels'+ '.npy',
            np.array(data))

    np.save(save_path + 'eyeRight_ROI_T1_56_6090_20_20pixels' + '.npy',
            np.array(data_one))
    np.save(save_path + 'eyeRight_ROI_T1_56_6090_30_30pixels' + '.npy',
            np.array(data_two))
