
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

def filter_signal(signals, ippg_per_second):
    HR_Min_HZ = 0.75
    HR_Max_HZ = 3.33
    RR_Min_HZ = 0.15
    RR_Max_HZ = 0.40
    FPS = int(35/ippg_per_second)
    # RR
    RR_filtered_signals = butterworth_filter(signals, RR_Min_HZ, RR_Max_HZ, FPS, order=5)
    # RR_filtered_signals_one = butterworth_filter(signals[0:int(len(signals)/2)], RR_Min_HZ, RR_Max_HZ,FPS, order=5)
    # RR_filtered_signals_two = butterworth_filter(signals[int(len(signals)/2):len(signals)], RR_Min_HZ, RR_Max_HZ, FPS, order=5)
    # print('RR_filtered_signals_one.shape = ', RR_filtered_signals_one.shape)
    # print('RR_filtered_signals_two.shape = ', RR_filtered_signals_two.shape)
    # RR_filtered_signals = np.concatenate((RR_filtered_signals_one, RR_filtered_signals_two), axis=0)
    # HR
    # HR_filtered_signals_one  = butterworth_filter(signals[0:int(len(signals)/2)], HR_Min_HZ, HR_Max_HZ, FPS, order=5)
    # HR_filtered_signals_two  = butterworth_filter(signals[int(len(signals)/2):len(signals)], HR_Min_HZ, HR_Max_HZ, FPS, order=5)
    # HR_filtered_signals = np.concatenate((HR_filtered_signals_one, HR_filtered_signals_two), axis=0)
    HR_filtered_signals  = butterworth_filter(signals, HR_Min_HZ, HR_Max_HZ, FPS, order=5)

    return RR_filtered_signals, HR_filtered_signals

def normalize(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))

def down_sample_ippg(data, ippg_per_second):
    data_arr = []
    for i in range(len(data)):
        if i % ippg_per_second==0:
            data_arr.append(data[i])
    return np.array(data_arr)

def count_num(data):
    count = 0
    for i in range(len(data)):
        if abs(data[i])<1:
            count = count+1
    print(count)

def get_time_domain(signal_arr): #(56, 3, 6090)
    HR_signal_arr = []
    RR_signal_arr = []
    signal_data = []
    ippg_per_second =3
    for i in range(signal_arr.shape[0]):
        for j in range(signal_arr.shape[1]):
            signal_pre =  down_sample_ippg(signal_arr[i, j, :], ippg_per_second)
            # signal_pre = normalize(signal_pre)
            print('signal_pre.shape=', signal_pre.shape)
            # 滤波
            RR_filtered_signals, HR_filtered_signals = filter_signal(signal_pre,ippg_per_second)
            # ippg 时域
            signal_data.append(signal_pre)
            # HR
            HR_signal_arr.append(HR_filtered_signals)
            # RR
            RR_signal_arr.append(RR_filtered_signals)
            print('signal_pre.shape = ',signal_pre.shape)
            # count_num(signal_pre)
            print('HR_filtered_signals.shape = ', HR_filtered_signals.shape)
            # count_num(HR_filtered_signals)
            print('RR_filtered_signals.shape= ', RR_filtered_signals.shape)
            # count_num(RR_filtered_signals)


    signal_data = np.array(signal_data)
    HR_signal_arr = np.array(HR_signal_arr)
    RR_signal_arr = np.array(RR_signal_arr)
    HR_signal_arr = np.reshape(HR_signal_arr, (signal_arr.shape[0], signal_arr.shape[1], int(signal_arr.shape[2]/ippg_per_second)))
    RR_signal_arr = np.reshape(RR_signal_arr, (signal_arr.shape[0], signal_arr.shape[1], int(signal_arr.shape[2]/ippg_per_second)))
    signal_data =  np.reshape(signal_data, (signal_arr.shape[0], signal_arr.shape[1], int(signal_arr.shape[2]/ippg_per_second)))
    print('HR_signal_arr.shape = ',HR_signal_arr.shape)
    print('RR_signal_arr.shape = ', RR_signal_arr.shape)
    print('signal_data.shape = ', signal_data.shape)
    signals = np.concatenate((signal_data, HR_signal_arr, RR_signal_arr), axis=1)
    print('signals.shape = ', signals.shape) #(56, 9, 6090)
    return signals

def train():
    # 6. forehead iPPG  单个ippg = (3, 6090)
    data_set_ippg_forehead_T1 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/ippg_forehead_T1_56_3_6090/'
        + 'ippg_forehead_T1_56_3_6090.npy')/255
    data_set_ippg_forehead_T2 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/ippg_forehead_T2_56_3_6090/'
        + 'ippg_forehead_T2_56_3_6090_notNAN.npy')/255
    # print('data_set_ippg_forehead_T1.shape = ', data_set_ippg_forehead_T1.shape)#(56, 3, 6090)
    # print('data_set_ippg_forehead_T2.shape = ', data_set_ippg_forehead_T2.shape)
    # 原始ippg： ippg_forehead_T1[:,0:3,:], HR_ippg: ippg_forehead_T1[:,3:6,:], RR_ippg: ippg_forehead_T1[:,6:9,:]
    ippg_forehead_T1 = get_time_domain(data_set_ippg_forehead_T1)  # (56, 9, 6090)
    np.save('./data_set_ippg_normalize/' + 'ippg_forehead_T1' + '.npy', ippg_forehead_T1)
    ippg_forehead_T2 = get_time_domain(data_set_ippg_forehead_T2)  # (56, 9, 6090)
    np.save('./data_set_ippg_normalize/' + 'ippg_forehead_T2' + '.npy', ippg_forehead_T2)
    #
    # data_set_ippg_forehead = np.concatenate((ippg_forehead_T1, ippg_forehead_T2), axis=0)
    # print('data_set_ippg_forehead.shape = ', data_set_ippg_forehead.shape)
    # np.save('./data_set_ippg/' + 'data_set_ippg_forehead' + '.npy', data_set_ippg_forehead)
    # print('####################################')
    # ippg_forehead = data_set_ippg_forehead.reshape(
    #     (data_set_ippg_forehead.shape[0], data_set_ippg_forehead.shape[1] * data_set_ippg_forehead.shape[2]))
    # print('data_set_ippg_forehead.shape = ', data_set_ippg_forehead.shape)  # (112, 9, 6090)
    # del data_set_ippg_forehead_T1, data_set_ippg_forehead_T2, ippg_forehead_T1, ippg_forehead_T2

    # # # 6. nose iPPG
    data_set_ippg_nose_T1 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/ippg_nose_T1_56_3_6090/'
        + 'ippg_nose_T1_56_3_6090.npy')/255
    data_set_ippg_nose_T2 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/ippg_nose_T2_56_3_6090/'
        + 'ippg_nose_T2_56_3_6090.npy')/255
    print('data_set_ippg_nose_T1.shape = ', data_set_ippg_nose_T1.shape)  # (56, 3, 6090)
    print('data_set_ippg_nose_T2.shape = ', data_set_ippg_nose_T2.shape)  # (56, 3, 6090)
    ippg_nose_T1 = get_time_domain(data_set_ippg_nose_T1)  # (56, 9, 6090)
    np.save('./data_set_ippg_normalize/' + 'ippg_nose_T1' + '.npy', ippg_nose_T1)
    ippg_nose_T2 = get_time_domain(data_set_ippg_nose_T2)  # (56, 9, 6090)
    np.save('./data_set_ippg_normalize/' + 'ippg_nose_T2' + '.npy', ippg_nose_T2)
    # data_set_ippg_nose = np.concatenate((ippg_nose_T1, ippg_nose_T2), axis=0)
    # np.save('./data_set_ippg/' + "data_set_ippg_nose" +'.npy',
    #         data_set_ippg_nose)
    # ippg_nose = data_set_ippg_nose.reshape(
    #     (data_set_ippg_nose.shape[0], data_set_ippg_nose.shape[1] * data_set_ippg_nose.shape[2]))
    # print('data_set_ippg_nose.shape = ', data_set_ippg_nose.shape)  # (112, 9, 6090)
    # del data_set_ippg_nose_T1, data_set_ippg_nose_T2, ippg_nose_T1, ippg_nose_T2

    # ippg
    # iPPG = np.concatenate((data_set_ippg_forehead, data_set_ippg_nose), axis=1)  # (112, 18, 6090)
    # iPPG = iPPG.transpose((0, 2, 1))  # (112, 6090, 18)
    # print('iPPG.shape = ', iPPG.shape)
    # iPPG = iPPG.reshape((iPPG.shape[0], iPPG.shape[1] * iPPG.shape[2]))
    # print('iPPG.shape = ', iPPG.shape)  # (112, 6090*18)
if __name__ == "__main__":
    # train constants
   train()