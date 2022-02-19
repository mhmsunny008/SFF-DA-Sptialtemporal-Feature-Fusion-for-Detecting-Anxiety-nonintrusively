
import numpy as np
import pandas as pd
import warnings
import xgboost as xgb
# import seaborn as sns
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
def butterworth_filter(data, low, high, sample_rate, order=5):
    from scipy import signal
    nyquist_rate = sample_rate * 1
    low /= nyquist_rate
    high /= nyquist_rate
    # print('nyquist_rate = ', nyquist_rate)
    # print('low = ', low)
    # print('high = ', high)
    b, a = signal.butter(N=order, Wn=[low, high], btype='bandpass')
    return signal.lfilter(b, a, data)
def filter_signal(signals):
    HR_Min_HZ = 0.75
    HR_Max_HZ = 3.33
    RR_Min_HZ = 0.15
    RR_Max_HZ = 0.40
    FPS = 35
    # RR
    RR_filtered_signals = butterworth_filter(signals, RR_Min_HZ, RR_Max_HZ,FPS, order=5)

    # HR
    HR_filtered_signals = butterworth_filter(signals, HR_Min_HZ, HR_Max_HZ, FPS, order=5)
    return RR_filtered_signals, HR_filtered_signals

def normalize(signals):
    normalized_signals = normalized_signals = (signals - np.min(signals)) / (np.max(signals)-np.min(signals))
    return normalized_signals
def detrend(signals, param_lambda):
    # https://blog.csdn.net/piaoxuezhong/article/details/79211586
    signal_len = len(signals)
    I = np.identity(signal_len)
    B = np.array([1, -2, 1])
    ones = np.ones((signal_len - 2, 1))
    multi = B * ones
    D2 = np.zeros((signal_len - 2, signal_len))
    for i in range(D2.shape[0]):
        D2[i, i:i + 3] = multi[i]
    tr_D2 = np.transpose(D2)
    multi_D2 = np.dot(tr_D2, D2)
    inverse = I - (np.linalg.inv(I + (multi_D2 * pow(param_lambda, 2))))
    detrend_signals = np.dot(inverse, signals)
    return detrend_signals
def preprocessing(signals):
    # 去趋势
    detrend_signals = detrend(signals, 100)
    detrend_signals = detrend_signals.flatten()
    # 标准化
    detrend_signals = normalize(detrend_signals)
    return detrend_signals

def get_time_domain(signal_arr): #(56, 3, 6090)

    HR_signal_arr = []
    RR_signal_arr = []
    signal_data = []
    for i in range(signal_arr.shape[0]):
        for j in range(signal_arr.shape[1]):
            signal =  normalize(signal_arr[i, j, :])
            signal_normalize = preprocessing(signal)
            # 滤波
            RR_filtered_signals, HR_filtered_signals = filter_signal(signal_normalize)
            # ippg 时域
            signal_data.append(signal)
            # HR
            HR_signal_arr.append(HR_filtered_signals)
            # RR
            RR_signal_arr.append(RR_filtered_signals)
            print('signal.shape = ',signal.shape)
            print('RR_filtered_signals.shape = ', RR_filtered_signals.shape)
            print('RR_filtered_signals.shape = ', RR_filtered_signals.shape)
    signal_data = np.array(signal_data)
    HR_signal_arr = np.array(HR_signal_arr)
    RR_signal_arr = np.array(RR_signal_arr)
    HR_signal_arr = np.reshape(HR_signal_arr, (signal_arr.shape[0], signal_arr.shape[1], signal_arr.shape[2]))
    RR_signal_arr = np.reshape(RR_signal_arr, (signal_arr.shape[0], signal_arr.shape[1], signal_arr.shape[2]))
    signal_data =  np.reshape(signal_data, (signal_arr.shape[0], signal_arr.shape[1], signal_arr.shape[2]))
    print('HR_signal_arr.shape = ',HR_signal_arr.shape)
    print('RR_signal_arr.shape = ', RR_signal_arr.shape)
    print('signal_data.shape = ', signal_data.shape)
    signals = np.concatenate((signal_data, HR_signal_arr, RR_signal_arr), axis=1)
    print('signals.shape = ', signals.shape) #(56, 9, 6090)
    return signals

def train():
    # # 6. forehead iPPG  单个ippg = (3, 6090)
    # data_set_ippg_forehead_T1 = np.load(
    #     r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/ippg_forehead_T1_56_3_6090/'
    #     + 'ippg_forehead_T1_56_3_6090.npy')
    # data_set_ippg_forehead_T2 = np.load(
    #     r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/ippg_forehead_T2_56_3_6090/'
    #     + 'ippg_forehead_T2_56_3_6090_notNAN.npy')
    # # print('data_set_ippg_forehead_T1.shape = ', data_set_ippg_forehead_T1.shape)#(56, 3, 6090)
    # # print('data_set_ippg_forehead_T2.shape = ', data_set_ippg_forehead_T2.shape)
    # # 原始ippg： ippg_forehead_T1[:,0:3,:], HR_ippg: ippg_forehead_T1[:,3:6,:], RR_ippg: ippg_forehead_T1[:,6:9,:]
    # ippg_forehead_T1 = get_time_domain(data_set_ippg_forehead_T1[0:2,:,:])  # (56, 9, 6090)
    # ippg_forehead_T2 = get_time_domain(data_set_ippg_forehead_T2[0:2,:,:])  # (56, 9, 6090)
    # data_set_ippg_forehead = np.concatenate((ippg_forehead_T1, ippg_forehead_T2), axis=0)
    # print('data_set_ippg_forehead.shape = ', data_set_ippg_forehead.shape)
    # # np.savetxt('./data_ippg/data_set_ippg_forehead.npy', data_set_ippg_forehead)
    # np.save('./data_ippg/' + 'data_set_ippg_forehead' + '.npy', data_set_ippg_forehead)
    # print('####################################')
    # ippg_forehead = data_set_ippg_forehead.reshape(
    #     (data_set_ippg_forehead.shape[0], data_set_ippg_forehead.shape[1] * data_set_ippg_forehead.shape[2]))
    # print('data_set_ippg_forehead.shape = ', data_set_ippg_forehead.shape)  # (112, 9, 6090)
    # del data_set_ippg_forehead_T1, data_set_ippg_forehead_T2, ippg_forehead_T1, ippg_forehead_T2

    # # 6. nose iPPG
    data_set_ippg_nose_T1 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/ippg_nose_T1_56_3_6090/'
        + 'ippg_nose_T1_56_3_6090.npy')
    data_set_ippg_nose_T2 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/ippg_nose_T2_56_3_6090/'
        + 'ippg_nose_T2_56_3_6090.npy')
    print('data_set_ippg_nose_T1.shape = ', data_set_ippg_nose_T1.shape)  # (56, 3, 6090)
    print('data_set_ippg_nose_T2.shape = ', data_set_ippg_nose_T2.shape)  # (56, 3, 6090)
    ippg_nose_T1 = get_time_domain(data_set_ippg_nose_T1[0:2,:,:])  # (56, 9, 6090)
    ippg_nose_T2 = get_time_domain(data_set_ippg_nose_T2[0:2,:,:])  # (56, 9, 6090)
    data_set_ippg_nose = np.concatenate((ippg_nose_T1, ippg_nose_T2), axis=0)
    np.save('./data_ippg/' + "data_set_ippg_nose" +'.npy',
            data_set_ippg_nose)
    ippg_nose = data_set_ippg_nose.reshape(
        (data_set_ippg_nose.shape[0], data_set_ippg_nose.shape[1] * data_set_ippg_nose.shape[2]))
    print('data_set_ippg_nose.shape = ', data_set_ippg_nose.shape)  # (112, 9, 6090)
    del data_set_ippg_nose_T1, data_set_ippg_nose_T2, ippg_nose_T1, ippg_nose_T2

    # ippg
    # iPPG = np.concatenate((data_set_ippg_forehead, data_set_ippg_nose), axis=1)  # (112, 18, 6090)
    # iPPG = iPPG.transpose((0, 2, 1))  # (112, 6090, 18)
    # print('iPPG.shape = ', iPPG.shape)
    # iPPG = iPPG.reshape((iPPG.shape[0], iPPG.shape[1] * iPPG.shape[2]))
    # print('iPPG.shape = ', iPPG.shape)  # (112, 6090*18)
if __name__ == "__main__":
    # train constants
   train()