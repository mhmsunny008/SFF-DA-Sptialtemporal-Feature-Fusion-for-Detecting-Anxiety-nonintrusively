# /home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-10-21-SVM-paper2/
# 3090环境mhmpy37
# https://www.omegaxyz.com/2018/01/12/python_svm/
# 数据集下载地址：http://archive.ics.uci.edu/ml/machine-learning-databases/iris/， 下载iris.data即可
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import warnings
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve
from inspect import signature
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from imblearn.metrics import geometric_mean_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.metrics import sensitivity_score, specificity_score, geometric_mean_score
from sklearn.metrics import accuracy_score,  roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import os
# warnings.filterwarnings('ignore')
#
# def load_data(filename):
#     data = pd.read_csv(filename)
#     data = data.values  # 转换成矩阵形式
#     x = data[:, 0:-1]
#     y = data[:, -1]
#     return x, y
def down_sample(label, data_set_mouthRIO_mod4_0):
    label_axniety = []
    data_mouth_anxiety = []
    label_no_anxiety = []
    data_mouth_no_anxiety = []
    count = 0
    for i in range(len(label)):
        if label[i] != 0:
            label_axniety.append(label[i])
            data_mouth_anxiety.append(data_set_mouthRIO_mod4_0[i])
        else:
            if count < 41:
                label_no_anxiety.append(label[i])
                data_mouth_no_anxiety.append(data_set_mouthRIO_mod4_0[i])
                count = count + 1
    label = np.concatenate((label_axniety, label_no_anxiety), axis=0)  # label标签拼接
    data = np.concatenate((data_mouth_anxiety, data_mouth_no_anxiety), axis=0)  # label标签拼接
    return np.array(data), np.array(label)

def data_pre(data_set):
    data_pre = []
    for i in range(data_set.shape[0]):
        for j in range(data_set.shape[1]):
            for m in range(data_set.shape[3]):
                distance = data_set[i, j, 0, m] * data_set[i, j, 0, m] + data_set[i, j, 1, m] * data_set[i, j, 1, m] +data_set[i, j, 2, m] * data_set[i, j, 2, m]
                data_pre.append(distance)
    data_pre = np.reshape(data_pre,(data_set.shape[0],data_set.shape[1],data_set.shape[3]))
    return np.array(data_pre)

def get_mouth_signal(data_set_mouthRIO): # (112, 2030, 40, 40, 3)
    print('data_set_mouthRIO.shape = ', data_set_mouthRIO.shape)
    mouth_signal_arr = []
    mouth_r = []
    mouth_g = []
    mouth_b = []
    for i in range(data_set_mouthRIO.shape[0]): #164

        for j in range(data_set_mouthRIO.shape[1]):  #100
            img = data_set_mouthRIO[i,j,:,:,:]
            # print('img.shape = ', img.shape)
            mouth_r.append(np.mean(img[:, :, 2]))
            mouth_g.append(np.mean(img[:, :, 1]))
            mouth_b.append(np.mean(img[:, :, 0]))
    mouth_r = np.array(mouth_r)
    mouth_g = np.array(mouth_g)
    mouth_b = np.array(mouth_b)
    mouth_signal_arr.append(mouth_r)
    mouth_signal_arr.append(mouth_g)
    mouth_signal_arr.append(mouth_b)
    mouth_signal_arr = np.array(mouth_signal_arr)
    print('mouth_signal_arr.shape = ', mouth_signal_arr.shape) # (3, 112*2030)
    print('mouth_r.shape = ', mouth_r.shape)  # (112*2030)
    print('mouth_g.shape = ', mouth_g.shape)
    print('mouth_b.shape = ', mouth_b.shape)
    mouth_signal_arr = mouth_signal_arr.reshape((data_set_mouthRIO.shape[0],data_set_mouthRIO.shape[4],data_set_mouthRIO.shape[1]))
    mouth_signal_arr = mouth_signal_arr.transpose((1,2,0)) # (164, 100, 3)
    mouth_signal_arr = mouth_signal_arr.reshape((data_set_mouthRIO.shape[0], data_set_mouthRIO.shape[1]*data_set_mouthRIO.shape[4]))
    print('mouth_signal_arr.shape = ', mouth_signal_arr.shape) # (112, 2030*3)
    return mouth_signal_arr

def get_AU_eye(data_AU,data_eye):
    print('data_AU.shape =', data_AU.shape) #  (82, 400, 17)
    print('data_eye.shape = ', data_eye.shape)  # (82, 400, 2)
    data_AU_0 = []
    data_AU_1 = []
    data_eye_0 = []
    data_eye_1 = []
    data_AU_arr = []
    data_eye_arr = []
    for i in range(data_AU.shape[0]):
        for j in range(data_AU.shape[1]):
            if j%4 == 0:
                data_AU_0.append(data_AU[i,j,:])
                data_eye_0.append(data_eye[i,j,:])
            if j%4 == 1:
                data_AU_1.append(data_AU[i,j,:])
                data_eye_1.append(data_eye[i, j,:])
    data_AU_0 = np.array(data_AU_0)
    data_AU_1 = np.array(data_AU_1)
    data_eye_0 = np.array(data_eye_0)
    data_eye_1 = np.array(data_eye_1)
    data_AU_arr.append(data_AU_0)
    data_AU_arr.append(data_AU_1)
    data_eye_arr.append(data_eye_0)
    data_eye_arr.append(data_eye_1)
    data_AU_arr = np.array(data_AU_arr)
    data_eye_arr = np.array(data_eye_arr)
    data_AU_arr = data_AU_arr.reshape((2,82,100,17))
    data_eye_arr = data_eye_arr.reshape((2,82,100,2))
    ##
    data_AU_arr = data_AU_arr.reshape((2*82, 100,17))
    data_eye_arr = data_eye_arr.reshape((2*82, 100,2))
    data_AU_arr = data_AU_arr.reshape((2 * 82, 100*17))
    data_eye_arr = data_eye_arr.reshape((2 * 82, 100*2))
    # print('data_AU_arr.shape = ',data_AU_arr.shape)     #  (164, 1700)
    # print('data_eye_arr.shape = ', data_eye_arr.shape)  #  (164, 200)
    return data_AU_arr, data_eye_arr
def concatenate_arr(x1, x2):
    x = []
    print('x1.shape = ', x1.shape)
    print('x2.shape = ', x2.shape)
    print('x1.shape[0] = ', x1.shape[0] )
    print('x1.shape[1] = ', x1.shape[1])
    print('x2.shape[0] = ', x2.shape[0] )
    print('x2.shape[1] = ', x2.shape[1])

    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            x.append(x1[i,j])
        for k in range(x2.shape[1]):
            x.append(x2[i, k])
    x = np.array(x)
    x = x.reshape((x1.shape[0], x1.shape[1]+x2.shape[1]))
    return x


def pre_data_label( data_set_mouth_signal_train, data_set_mouth_signal_test,
                    # data_set_AU_train, data_set_AU_test,
                    data_set_eye_train, data_set_eye_test,
                    data_set_loc_train, data_set_loc_test,
                    # data_set_iPPG_train, data_set_iPPG_test,
                    ):
    print('data_set_mouth_signal_train.shape = ', data_set_mouth_signal_train.shape)
    # print('data_set_AU_train.shape = ', data_set_AU_train.shape)
    print('data_set_eye_train.shape = ', data_set_eye_train.shape)
    print('data_set_loc_train.shape = ', data_set_loc_train.shape)
    # print('data_set_iPPG_train.shape = ', data_set_iPPG_train.shape)
    x_train = []
    x_test = []
    # 训练集数据拼接
    # x_train= concatenate_arr(data_set_mouth_signal_train,data_set_AU_train)
    x_train = concatenate_arr(data_set_mouth_signal_train, data_set_eye_train)
    x_train = concatenate_arr(x_train, data_set_loc_train)
    # x_train = concatenate_arr(x_train, data_set_iPPG_train)

    # 测试集数据拼接
    # x_test = concatenate_arr(data_set_mouth_signal_test, data_set_AU_test)
    x_test = concatenate_arr(data_set_mouth_signal_test, data_set_eye_test)
    x_test = concatenate_arr(x_test, data_set_loc_test)
    # x_test = concatenate_arr(x_test, data_set_iPPG_test)


    # x_train = np.array(x_train)
    # x_test = np.array(x_test)
    print("x_train.shape = ", x_train.shape)
    print("x_test.shape= ",x_test.shape)
    return x_train, x_test


def onehot(label_set):
    label_index_anxiety = []
    label_index_anxiety_free = []
    for i in range(len(label_set)):
        if label_set[i] != 0:
            label_set[i] = 1
            label_index_anxiety.append(i)
        else:
            label_set[i] = 0
            label_index_anxiety_free.append(i)

    return label_set, label_index_anxiety, label_index_anxiety_free

def get_AU_eyegaze(str_T, frame_num):
    path = "/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/data_set_AU_UBFC_Phys_T1_T2/"
    print('path = ', path)
    dirs = os.listdir(path)  # /home/som/8T/DataSets/ubfc_phys/video/s16/
    count = 0
    data_eyegaze_arr = []
    data_head_pos_arr = []
    data_au_arr = []
    for file in dirs:
        if file[-6:] == str_T:
            print(file)
            data_path = path + file
            print(data_path)
            count = count + 1
            data = pd.read_csv(data_path, header=1)
            print('data.shape = ', data.shape)
            data_eyegaze = data.values[0:frame_num, 5:7]
            data_head_pos = data.values[0:frame_num, 7:10]
            data_au = data.values[0:frame_num, 10:27]
            print('data_eyegaze.shape = ', data_eyegaze.shape)
            print('data_head_pos.shape = ', data_head_pos.shape)
            print('data_au.shape = ', data_au.shape)
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
    print(count)
    return data_au_arr, data_eyegaze_arr

def filter_signal_infrared(signals, Min_HZ,Max_HZ, FPS):
    filtered_signals = butterworth_filter(signals, Min_HZ, Max_HZ, FPS, order=5)
    return filtered_signals

def butterworth_filter(data, low, high, sample_rate, order=2):
    from scipy import signal
    nyquist_rate = sample_rate * 0.5
    low /= nyquist_rate
    high /= nyquist_rate
    b, a = signal.butter(N=order, Wn=[low, high], btype='bandpass')
    return signal.lfilter(b, a, data)

def get_time_domain(signal_arr): #(56, 3, 6090)
    HR_Min_HZ = 0.75
    HR_Max_HZ = 3.33
    RR_Min_HZ = 0.15
    RR_Max_HZ = 0.40
    FPS = 35
    HR_signal_arr = []
    RR_signal_arr = []
    for i in range(signal_arr.shape[0]):
        for j in range(signal_arr.shape[1]):
            # HR
            HR_signal = filter_signal_infrared(signal_arr[i,j,:], HR_Min_HZ, HR_Max_HZ, FPS)
            HR_signal_arr.append(HR_signal)
            # RR
            RR_signal = filter_signal_infrared(signal_arr[i, j, :], RR_Min_HZ, RR_Max_HZ, FPS)
            RR_signal_arr.append(RR_signal)
    HR_signal_arr = np.reshape(HR_signal_arr, (signal_arr.shape[0], signal_arr.shape[1], signal_arr.shape[2]))
    RR_signal_arr = np.reshape(RR_signal_arr, (signal_arr.shape[0], signal_arr.shape[1], signal_arr.shape[2]))
    print('HR_signal_arr.shape = ',HR_signal_arr.shape)
    print('RR_signal_arr.shape = ', RR_signal_arr.shape)
    signals = np.concatenate((signal_arr, HR_signal_arr, RR_signal_arr), axis=1)
    print('signals.shape = ', signals.shape)
    return signals


def img_dowm_sample(data):
    img_num_per_second = 3
    data_arr = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if j%img_num_per_second==0:
                data_arr.append(data[i,j])
    data_arr = np.reshape(data_arr,(data.shape[0], int(data.shape[1]/img_num_per_second),data.shape[2],data.shape[3],data.shape[4]))
    del data
    return data_arr

def train():
    no_epochs = 50  # 5000 originally
    # no_videos_by_class = 200
    batch_size = 2
    # learning_rate = 1e-4  # IMG
    learning_rate = 1e-3  #
    no_classes = 2
    # classes = 1
    ##
    test_size = 0.2
    random_state = 3
    batch_size = 2
    # label
    num_volunteer = 56
    label = []
    for i in range(num_volunteer):  # label: stress-free
        label.append(0)
    for i in range(num_volunteer):  # label: stress
        label.append(1)
    label = np.array(label)
    print('label.shape = ', label.shape)  # (112,)

    # 1. 嘴巴
    img_num_tatal = 6090
    data_set_mouthRIO_T1 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/mouth_ROI_T1_56_6090_40_40pixels/'
        + 'mouth_ROI_T1_56_6090_40_40pixels.npy')/ 255
    data_set_mouthRIO_T2 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/mouth_ROI_T2_56_6090_40_40pixels/'
        + 'mouth_ROI_T2_56_6090_40_40pixels.npy')/ 255
    # data_set_mouthRIO_T1 = data_set_mouthRIO_T1[:, 0:img_num_tatal, :, :, :]
    # data_set_mouthRIO_T2 = data_set_mouthRIO_T2[:, 0:img_num_tatal, :, :, :]
    data_set_mouthRIO_T1 = img_dowm_sample(data_set_mouthRIO_T1)
    data_set_mouthRIO_T2 = img_dowm_sample(data_set_mouthRIO_T2)
    data_set_mouthRIO = np.concatenate((data_set_mouthRIO_T1, data_set_mouthRIO_T2), axis=0) #(112, 6090, 40, 40, 3)
    # data_set_mouthRIO = data_set_mouthRIO.transpose((0, 4, 2, 3, 1))
    print('data_set_mouthRIO_T1.shape = ', data_set_mouthRIO_T1.shape)
    print('data_set_mouthRIO_T2.shape = ', data_set_mouthRIO_T2.shape)
    print('data_set_mouthRIO.shape = ', data_set_mouthRIO.shape)
    mouth_signal_arr = get_mouth_signal(data_set_mouthRIO)
    print('mouth_signal_arr.shape = ', mouth_signal_arr.shape)
    del data_set_mouthRIO_T1, data_set_mouthRIO_T2

    # 2. AU   and   3. eyegaze
    str_T1 = 'T1.csv'
    str_T2 = 'T2.csv'
    frame_num = 6090
    data_AU_T1, data_eyegaze_T1 = get_AU_eyegaze(str_T1,frame_num)
    data_AU_T2, data_eyegaze_T2 = get_AU_eyegaze(str_T2, frame_num)
    data_AU_arr = np.concatenate((data_AU_T1, data_AU_T2), axis=0)
    data_eyegaze_arr = np.concatenate((data_eyegaze_T1, data_eyegaze_T2), axis=0)
    print('data_AU_arr.shape =', data_AU_arr.shape) #  (112, 6090, 17)
    print('data_eyegaze_arr.shape =', data_eyegaze_arr.shape)  #  (112, 6090, 2)
    data_AU_arr = np.reshape(data_AU_arr,(data_AU_arr.shape[0], data_AU_arr.shape[1]*data_AU_arr.shape[2]))
    data_eyegaze_arr = np.reshape(data_eyegaze_arr, (data_eyegaze_arr.shape[0], data_eyegaze_arr.shape[1] * data_eyegaze_arr.shape[2]))
    print('data_AU_arr.shape =', data_AU_arr.shape)  # (112, 6090*17)
    print('data_eyegaze_arr.shape =', data_eyegaze_arr.shape)  # (112, 6090*2)
    # 5. 头部位姿
    loc_arr_T1 = np.load(
        '/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/loc_T1_56_6090_468/'
        + 'loc_distan_T1_56_6090_468.npy')
    loc_arr_T2 = np.load(
        '/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/loc_T2_56_6090_468/'
        + 'loc_distan_T2_56_6090_468.npy')
    print('loc_arr_T1.shape = ', loc_arr_T1.shape)  # (56, 6090, 468)
    print('loc_arr_T2.shape = ', loc_arr_T2.shape)  # (56, 6090, 468)
    location = np.concatenate((loc_arr_T1, loc_arr_T2), axis=0)
    print('location.shape = ', location.shape)  # (112, 6090, 468)
    location = location.reshape((location.shape[0], location.shape[1] * location.shape[2]))
    print('location.shape =', location.shape)  # (112, 6090*468)
    del loc_arr_T1, loc_arr_T2

    # 6. forehead iPPG  单个ippg = (3, 6090)
    ippg_forehead_T1 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/data_set_ippg/'
        + 'ippg_forehead_T1.npy')  # (56, 9, 6090)
    ippg_forehead_T2 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/data_set_ippg/'
        + 'ippg_forehead_T2.npy')  # (56, 9, 6090)
    data_set_ippg_forehead = np.concatenate((ippg_forehead_T1, ippg_forehead_T2), axis=0)
    ippg_forehead = data_set_ippg_forehead.reshape(
        (data_set_ippg_forehead.shape[0], data_set_ippg_forehead.shape[1] * data_set_ippg_forehead.shape[2]))
    print('data_set_ippg_forehead.shape = ', data_set_ippg_forehead.shape)  # (112, 9, 6090)

    del ippg_forehead_T1, ippg_forehead_T2

    # # 6. nose iPPG
    ippg_nose_T1 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/data_set_ippg/'
        + 'ippg_nose_T1.npy')  # (56, 9, 2030)
    ippg_nose_T2 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/data_set_ippg/'
        + 'ippg_nose_T2.npy')  # (56, 9, 2030)
    data_set_ippg_nose = np.concatenate((ippg_nose_T1, ippg_nose_T2), axis=0)
    ippg_nose = data_set_ippg_nose.reshape(
        (data_set_ippg_nose.shape[0], data_set_ippg_nose.shape[1] * data_set_ippg_nose.shape[2]))
    print('data_set_ippg_nose.shape = ', data_set_ippg_nose.shape)  # (112, 9, 2030)
    del ippg_nose_T1, ippg_nose_T2

    # ippg
    iPPG = np.concatenate((data_set_ippg_forehead, data_set_ippg_nose), axis=1)  # (112, 18, 2030)
    iPPG = iPPG.transpose((0, 2, 1))  # (112, 2030, 18)
    print('iPPG.shape = ', iPPG.shape)
    iPPG = iPPG.reshape((iPPG.shape[0], iPPG.shape[1] * iPPG.shape[2]))
    print('iPPG.shape = ', iPPG.shape)  # (112, 2030*18)


    # 标签
    label_set, label_index_anxiety, label_index_anxiety_free = onehot(label)
    y_labels_train, y_labels_val = train_test_split(label_set, test_size=test_size,
                                                    random_state=random_state)
    y_labels_train = np.array(y_labels_train)
    y_labels_val = np.array(y_labels_val)

    # mouth_signal
    data_set_mouth_signal_train, data_set_mouth_signal_test = train_test_split(mouth_signal_arr, test_size=test_size,
                                                           random_state=random_state)
    # AU
    data_set_AU_train, data_set_AU_test = train_test_split(data_AU_arr, test_size=test_size,
                                                           random_state=random_state)

    # eye
    data_set_eye_train, data_set_eye_test = train_test_split(data_eyegaze_arr, test_size=test_size,
                                                           random_state=random_state)

    # iPPG
    data_set_iPPG_train, data_set_iPPG_test = train_test_split(iPPG, test_size=test_size,
                                                           random_state=random_state)

    # location
    data_set_loc_train, data_set_loc_test = train_test_split(location, test_size=test_size,
                                                               random_state=random_state)
    y_train = y_labels_train.reshape((y_labels_train.shape[0],1))
    y_test = y_labels_val.reshape((y_labels_val.shape[0],1))
    print('y_train.shape = ', y_train.shape)
    print('y_test.shape = ', y_test.shape)
    # x_train, x_test =  data_set_mouth_signal_train, data_set_mouth_signal_test
    # x_train, x_test = data_set_AU_train, data_set_AU_test
    # x_train, x_test = data_set_eye_train, data_set_eye_test
    # x_train, x_test = data_set_loc_train, data_set_loc_test
    # x_train, x_test = data_set_iPPG_train, data_set_iPPG_test
    x_train,x_test = pre_data_label(
                    data_set_mouth_signal_train, data_set_mouth_signal_test,
                    # data_set_AU_train, data_set_AU_test,
                    data_set_eye_train, data_set_eye_test,
                    data_set_loc_train, data_set_loc_test,
                    # data_set_iPPG_train, data_set_iPPG_test,
                    )

    # ------- RandomForest --------#
    # from sklearn.ensemble import RandomForestClassifier  # 随机森林
    # model_RandomForest = RandomForestClassifier()  # 实例化
    # # model_RandomForest = model_RandomForest.fit(X_train, y_train)  # 用训练集数据训练模型
    # model_RandomForest.fit(x_train, y_train)
    # test_predict_label = model_RandomForest.predict_proba(x_test)
    # preds = model_RandomForest.predict(x_test)  # 导入测试集，从接口中调用需要的信息
    # ------- RandomForest --------#

    # ------- LogisticRegression --------#
    from sklearn.linear_model import LogisticRegression  # 逻辑回归
    model_log_reg = LogisticRegression()
    model_log_reg.fit(x_train, y_train)
    test_predict_label = model_log_reg.predict_proba(x_test)
    preds = model_log_reg.predict(x_test)
    # ------- LogisticRegression --------#
    # 首先通过fit来对训练样本和训练样本标签进行训练得到模型，然后通过decision_function来获得模型对于测试样本集预测的标签集
    print('test_predict_label = ', test_predict_label)
    print('test_predict_label.shape = ', test_predict_label.shape)

    # Compute ROC curve and ROC area for each class#计算tp,fp
    # 通过测试样本输入的标签集和模型预测的标签集进行比对，得到fp,tp,不同的fp,tp是算法通过一定的规则改变阈值获得的
    fpr, tpr, threshold = roc_curve(y_test, test_predict_label[:, 1])  ###计算真正率和假正率
    # fpr, tpr, threshold = roc_curve(y_test, test_predict_label)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值，auc就是曲线包围的面积，越大越好
    ##  log_reg
    # log_reg = LogisticRegression()
    # log_reg.fit(x_train, y_labels_train)
    # preds = log_reg.predict(x_test)
    # print("log_reg.score(x_train, y_train)= ", log_reg.score(x_train, y_train))  # 精度
    # print("log_reg.score(x_test, y_test) =  ", log_reg.score(x_test, y_test))  # 精度
    # print('preds = ', preds)

    # 模型性能相关指标
    y_true = y_test
    y_pred_probability = test_predict_label[:, 1]
    # y_pred_probability = test_predict_label
    y_pred = preds
    acc = accuracy_score(y_test, preds)
    # auc = roc_auc_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall1 = recall_score(y_test, preds)
    sensitivity = sensitivity_score(y_test, preds)
    specificity = specificity_score(y_test, preds)
    fscore1 = f1_score(y_test, preds, average='weighted')
    gmean = geometric_mean_score(y_test, preds)  # 几何平均
    mat = confusion_matrix(y_test, preds)
    print("Accuracy: ", acc)
    print("ROC_Auc: ", roc_auc)
    print("Precision:", precision)
    print("Recall:", recall1)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("fscore1: ", fscore1)
    print('G-mean:', gmean)
    print("Confusion matrix: \n", mat)
    print('Overall report: \n', classification_report(y_test, preds))
    # # save result
    save_name = '_' + 'epochs' + str(no_epochs) + '_' + 'mouth' + '_' + 'eyeLeft' + '_' + 'eyeRight' + '_' + 'location'
    # + '_' + 'mouth' \
    # + '_' + 'eyeLeft' + '_' + 'eyeRight' \
    # + '_' + 'location'\
    # + '_' + 'forehead'\
    # + '_' + 'iPPG' # 额头提取iPPG
    # save result
    data_result = []
    data_result.append([acc, roc_auc, precision, recall1, sensitivity, fscore1, specificity])
    # np.savetxt('./data_result_RandomForest_40p_40pixel/RandomForest_fpr' + save_name + '.txt', fpr)
    # np.savetxt('./data_result_RandomForest_40p_40pixel/RandomForest_tpr' + save_name + '.txt', tpr)
    # np.savetxt('./data_result_RandomForest_40p_40pixel/RandomForest_data_result' + save_name + '.txt', data_result)
    # np.savetxt('./data_result_RandomForest_40p_40pixel/RandomForest_y_test' + save_name + '.txt', y_true)
    # np.savetxt('./data_result_RandomForest_40p_40pixel/RandomForest_y_pred_probability' + save_name + '.txt',
    #            y_pred_probability)
    # np.savetxt('./data_result_RandomForest_40p_40pixel/RandomForest_y_pred' + save_name + '.txt', y_pred)

    np.savetxt('./data_result_LogisticRegression_40p_40pixel/LogisticRegression_fpr' + save_name + '.txt', fpr)
    np.savetxt('./data_result_LogisticRegression_40p_40pixel/LogisticRegression_tpr' + save_name + '.txt', tpr)
    np.savetxt('./data_result_LogisticRegression_40p_40pixel/LogisticRegression_data_result' + save_name + '.txt', data_result)
    np.savetxt('./data_result_LogisticRegression_40p_40pixel/LogisticRegression_y_test' + save_name + '.txt', y_true)
    np.savetxt('./data_result_LogisticRegression_40p_40pixel/LogisticRegression_y_pred_probability' + save_name + '.txt',
               y_pred_probability)
    np.savetxt('./data_result_LogisticRegression_40p_40pixel/LogisticRegression_y_pred' + save_name + '.txt', y_pred)



if __name__ == "__main__":
    # train constants
   train()