import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import update_lr, plot_losses
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
import os
from imblearn.metrics import sensitivity_score, specificity_score, geometric_mean_score
from sklearn.metrics import accuracy_score,  roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc
import tensorflow as tf
from sklearn.metrics import roc_curve
from torchsummary import summary
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import *
import tensorflow.keras
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import metrics
import random
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print('torch.cuda.device_count()=', torch.cuda.device_count())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

def validate(device, batch_size, model, criterion,
             y_labels_val,
             data_set_mouthRIO_test,
             data_set_eyeLeftRIO_test, data_set_eyeRightRIO_test,
             data_set_xyz_test,
             # data_set_iPPG_test,
             training_proc_avg, test_proc_avg, last=False):
    # Test the model (validation set)
    model.eval()
    # sia_network.train()
    gen_signals_val = data_set_xyz_test
    y_labels_train  = y_labels_val
    ##
    model = model.to(device)
    total_loss = 0.0
    accuracy = 0
    y_true = []
    y_pred = []
    y_pred_probability = []
    y_pred_probability_arr = []
    with torch.no_grad():
        total_step = len(gen_signals_val) // batch_size

        for i in range(total_step):
            if i < total_step - 1:
                target = torch.from_numpy(y_labels_train[i * batch_size:(i + 1) * batch_size]).to(device).float()
                input_shape_mouth = torch.from_numpy(data_set_mouthRIO_test[i * batch_size:(i + 1) * batch_size]).to(device).float()
                input_shape_eyeLeft = torch.from_numpy(data_set_eyeLeftRIO_test[i * batch_size:(i + 1) * batch_size]).to(device).float()
                input_shape_eyeRight = torch.from_numpy(data_set_eyeRightRIO_test[i * batch_size:(i + 1) * batch_size]).to(device).float()
                input_shape_xyz = torch.from_numpy(data_set_xyz_test[i * batch_size:(i + 1) * batch_size]).to(device).float()
                # input_shape_iPPG = torch.from_numpy(data_set_iPPG_test[i * batch_size:(i + 1) * batch_size]).to(device).float()
            else:
                target = torch.from_numpy(y_labels_train[i * batch_size:-1]).to(device).to(device).float()
                input_shape_mouth = torch.from_numpy(data_set_mouthRIO_test[i * batch_size:-1]).to(device).float()
                input_shape_eyeLeft = torch.from_numpy(data_set_eyeLeftRIO_test[i * batch_size:-1]).to(device).float()
                input_shape_eyeRight = torch.from_numpy(data_set_eyeRightRIO_test[i * batch_size:-1]).to(device).float()
                input_shape_xyz = torch.from_numpy(data_set_xyz_test[i * batch_size:-1]).to(device).float()
                # input_shape_iPPG = torch.from_numpy(data_set_iPPG_test[i * batch_size:-1]).to(device).float()

            # Forward pass
            out = model(
                input_shape_mouth,
                input_shape_eyeLeft, input_shape_eyeRight,
                input_shape_xyz,
                # input_shape_iPPG,
            )

            # out_feature_fc_two = model_ConvNet3D_two(
            #     # input_shape_imgface,
            #     # input_shape_xyz,
            #     input_shape_eTou,
            #     input_shape_mouth,
            #     input_shape_eyeLeft, input_shape_eyeRight,
            #     # input_shape_eyebrowLeft, input_shape_eyebrowRight,
            #     # input_shape_nose,
            # )
            # out_sia_network = sia_network(out_feature_fc, out_feature_fc_two)
            # print('target= ', target)
            # print('out = ', out)
            loss = criterion(out, target)
            # Loss = loss+ out_sia_network



            # loss = F.cross_entropy(out, target)
            total_loss += loss.item()
            accuracy += (torch.argmax(out, dim=1) == target).sum().item()
            y_true.extend(torch.argmax(target, dim=1).cpu().numpy())
            y_pred.extend(torch.argmax(out, dim=1).cpu().numpy())
            y_pred_probability.extend(
                torch.nn.functional.softmax(out, dim=1).cpu().detach().numpy())  # 计算softmax，即该文本属于各类的概率

    y_pred_probability = np.array(y_pred_probability)
    # print('y_pred_probability  = ', y_pred_probability)
    # print('##############')
    y_pred_probability = y_pred_probability[:, 1]
    # print('y_pred_probability  = ', y_pred_probability)

    acc = accuracy_score(y_true, y_pred)
    print('acc = ', acc)
    return y_pred, y_true, y_pred_probability

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

def binary_onehot(y_labels_train):
    y_labels_train_arr = []
    for i in range(len(y_labels_train)):
        if y_labels_train[i] != 0:
            y_labels_train_arr.append([1,0])
        else:
            y_labels_train_arr.append([0,1])
    return np.array(y_labels_train_arr)
def data_pre(data_set):
    data_pre = []
    for i in range(data_set.shape[0]):
        for j in range(data_set.shape[1]):
            for m in range(data_set.shape[3]):
                distance = data_set[i, j, 0, m] * data_set[i, j, 0, m] + data_set[i, j, 1, m] * data_set[i, j, 1, m] +data_set[i, j, 2, m] * data_set[i, j, 2, m]
                data_pre.append(distance)
    data_pre = np.reshape(data_pre,(data_set.shape[0],data_set.shape[1],data_set.shape[3]))
    return np.array(data_pre)


# 3D CNN
# this assumes input to be of shape
# no_samples_batch, img_channels, no_frames, img_width, img_height

# to accept different shape would need to adjust
# what comes before layer 1

# one option is to resize the last two dimensions (height/width)
# and then reduce/increase number of frames to 60 but that last option
# seems bad since it kills the temporal consistency

# another more versatile option is to change all the
# constant values such as 32, 58, 20 and so on for values
# calculated based on conv kernel formulas so they become variables depending
# on the initial input size
import torch.nn as nn

class ConvNet3D_one(nn.Module):
    def __init__(self,
                 num_classes,
                 fc_neurons=128):
        super(ConvNet3D_one, self).__init__()
        self.fc_neurons = fc_neurons
        self.num_classes = num_classes
        ## 定义网络
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(1, 11, 11), stride=(1, 2, 2),
                               padding=(0, 5, 5), padding_mode='replicate')
        self.relu1 = nn.ReLU(True)
        self.bn1 = nn.BatchNorm3d(num_features=64)
        # 网络的第一层加入注意力机制
        # features_num = 64
        # self.ca = ChannelAttention(features_num)
        # self.sa = SpatialAttention()
        ##
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=96, kernel_size=(1, 5, 5), stride=(1, 2, 2),
                               padding=(0, 2, 2), padding_mode='replicate')
        self.relu2 = nn.ReLU(True)
        self.bn2 = nn.BatchNorm3d(num_features=96)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        self.conv3 = nn.Conv3d(in_channels=96, out_channels=128, kernel_size=(1, 3, 3), stride=(1, 1, 1),
                               padding=(0, 1, 1), padding_mode='replicate')
        self.relu3 = nn.ReLU(True)
        self.bn3 = nn.BatchNorm3d(num_features=128)
        self.conv4 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(1, 3, 3), stride=(1, 1, 1),
                               padding=(0, 1, 1), padding_mode='replicate')
        self.relu4 = nn.ReLU(True)
        self.bn4 = nn.BatchNorm3d(num_features=64)
        self.conv5 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=(1, 3, 3), stride=(1, 1, 1),
                               padding=(0, 1, 1), padding_mode='replicate')
        self.relu5 = nn.ReLU(True)
        self.bn5 = nn.BatchNorm3d(num_features=32)
        # 网络的卷积层的最后一层加入注意力机制
        # self.inplanes = 32
        # self.ca1 = ChannelAttention(self.inplanes)
        # self.sa1 = SpatialAttention()
        ###
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 4, 2), ceil_mode=False)
        self.n_units = 128
        self.fc = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(6400, 1024),
            # nn.ReLU(True),
            # nn.Sigmoid(),
            # nn.Linear(1024, 512),
            # nn.Linear(512, 256),
            # nn.Linear(256, 128),
            # nn.Linear(128, num_classes),
            ########
            # nn.Linear(32*40*1*93,  self.n_units),# 3000张
            # nn.Linear(32 * 40 * 1 * 190, self.n_units), #6090张
            nn.Linear(32 * 40 * 1 * 95, self.n_units),  # 6090/2张
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(406980,  self.n_units),
        )

        ###  ippg

        self.n_inputs_iPPG = 18
        self.rnn_iPPG = nn.LSTM(
            input_size=self.n_inputs_iPPG,
            hidden_size= self.n_units,
            num_layers=1,
            batch_first=True,
        )

        ## LSTM
        # self.n_units = 128
        self.n_inputs = 468  # 3DCNN输出的特征大小.shape[2] = 200
        self.rnn = nn.LSTM(
            input_size=self.n_inputs,
            hidden_size=self.n_units,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(self.n_units, self.num_classes)

        # self.fc1 = nn.Sequential(nn.Linear(2*2, self.num_classes))
        num_features = 4 # 根据输入特征的个数进行修改
        self.fc_sum = nn.Sequential(nn.Linear(self.n_units * num_features, self.num_classes),
                                    nn.Sigmoid(), # 和nn.BCELoss()配合使用，详见https://blog.csdn.net/weixin_40522801/article/details/106616564
                                    )
        # self.fc_sum = nn.Sequential(nn.Linear(64 * num_features, self.num_classes),
        #                             nn.Sigmoid(),
        #                             # 和nn.BCELoss()配合使用，详见https://blog.csdn.net/weixin_40522801/article/details/106616564
        #                             )

    def feature_extract_3DCNN(self, x):
        # print('x.shape = ', x.shape)
        out = self.conv1(x)
        # out = self.relu1(out)
        out = F.relu(out)
        # print('out_conv1.shape = ', out.shape)
        out = self.bn1(out)
        # print('out_bn1.shape = ', out.shape)
        ## 注意力机制：网络的第一层加入注意力机制
        # out = self.ca(out) * out
        # out = self.sa(out) * out
        ##
        out = self.pool1(out)
        # print('out_pool1.shape = ', out.shape)
        out = self.conv2(out)
        # out = self.relu2(out)
        out = F.relu(out)
        # print('out_conv2.shape = ', out.shape)
        out = self.bn2(out)
        # print('out_bn2.shape = ', out.shape)
        out = self.pool2(out)
        # print('out_pool2.shape = ', out.shape)
        out = self.conv3(out)
        # out = self.relu3(out)
        out = F.relu(out)
        # print('out_conv3.shape = ', out.shape)
        out = self.bn3(out)
        # print('out_bn3.shape = ', out.shape)
        out = self.conv4(out)
        # out = self.relu4(out)
        out = F.relu(out)
        # print('out_conv4.shape = ', out.shape)
        out = self.bn4(out)
        # print('out_bn4.shape = ', out.shape)
        out = self.conv5(out)
        # out = self.relu4(out)
        out = F.relu(out)
        # print('out_conv5.shape = ', out.shape)
        out = self.bn5(out)
        # print('out_bn5.shape = ', out.shape)
        # 网络的卷积层的最后一层加入注意力机制
        # out = self.ca1(out) * out
        # out = self.sa1(out) * out
        ##
        out = self.pool3(out)
        # print('out.shape = ', out.shape)
        # out_pool3 = out.transpose((0, 1, 3, 2, 4))  # 数据维度变换
        # print('out_pool3.shape = ', out_pool3.shape)
        # print('out.shape = ', out.shape)
        out = self.fc(out)
        return out

    def feature_extract_LSTM(self, x):
        # print('x.shape= ', x.shape)
        x = x.to(device).float()
        r_out, (h_n, h_c) = self.rnn(x, None)
        # out = self.out(r_out[:, -1, :])
        out = r_out[:, -1, :]
        return out

    def feature_extract_iPPG(self, x):
        # print('x.shape= ', x.shape)
        x = x.to(device).float()
        r_out, (h_n, h_c) = self.rnn_iPPG(x, None)
        # out = self.out(r_out[:, -1, :])
        out = r_out[:, -1, :]
        return out

    def feature_extract_loc(self, x):
        features_arr = []
        # for i in range(data.shape[0]):
        #     x= data[i]
        # print('x.shape = ', x.shape)
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        # print('out.shape = ', out.shape)

        for i in range(out.shape[0]):
            feature = out[i]
            # 降维：拉成一维向量
            feature = torch.reshape(feature, (-1,))
            # print('feature.shape = ', feature.shape)
            feature = feature.cpu().detach().numpy()
            features_arr.append(feature)

        features_arr = np.array(features_arr)
        # print('features_arr.shape = ', features_arr.shape)
        #     features_arr.append(out)
        # features_arr = np.array(features_arr)
        # print('features_arr.shape = ', features_arr.shape)
        features_arr = self.fc_loc(torch.from_numpy(features_arr).to(device))
        return features_arr
    def forward(self,
                  input_shape_mouth,
                  input_shape_eyeLeft, input_shape_eyeRight,
                input_shape_xyz,
                # input_shape_iPPG,
                ):
        # print('input_shape_mouth.shape', input_shape_mouth.shape)
        # print('input_shape_eyeLeft.shape', input_shape_eyeLeft.shape)
        # print('input_shape_eyeRight.shape', input_shape_eyeRight.shape)
        # print('input_shape_eTou.shape', input_shape_eTou.shape)
        # print('input_shape_xyz.shape', input_shape_xyz.shape)

        features_mouth = self.feature_extract_3DCNN(input_shape_mouth)
        features_eyeLeft = self.feature_extract_3DCNN(input_shape_eyeLeft)
        features_eyeRight = self.feature_extract_3DCNN(input_shape_eyeRight)
        features_xyz = self.feature_extract_LSTM(input_shape_xyz)
        # features_iPPG = self.feature_extract_iPPG(input_shape_iPPG)
        # print('features_mouth.shape', features_mouth.shape)
        # print('features_eyeLeft.shape', features_eyeLeft.shape)
        # print('features_eyeRight.shape', features_eyeRight.shape)
        # print('features_xyz.shape', features_xyz.shape)
        # print('features_iPPG.shape', features_iPPG.shape)
        out = torch.cat((
            features_mouth,
            features_eyeLeft, features_eyeRight,
            features_xyz,
            # features_iPPG,
            ), 1)  # 在 1 维(横向)进行拼接
        out = self.fc_sum(out)
        return out
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
    data_arr = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if j%2==0:
                data_arr.append(data[i,j])
    data_arr = np.reshape(data_arr,(data.shape[0], int(data.shape[1]/2),40,40,3))
    del data
    return data_arr

def train():
    # train constants
    no_epochs = 50  # 5000 originally
    batch_size = 2
    # learning_rate = 1e-5  # img
    learning_rate = 1e-4  # img
    # learning_rate = 1e-3  # ippg
    ##
    test_size = 0.2
    random_state = 3
    batch_size = 2

    # 标签
    num_volunteer = 56
    label = []
    for i in range(num_volunteer):  # label: stress-free
        label.append(0)
    for i in range(num_volunteer):  # label: stress
        label.append(1)
    label = np.array(label)
    print('label.shape = ', label.shape)  # (217,)
    # 一、 行为数据：眼睛、嘴巴、头部位姿
    # 1. 嘴巴
    img_num_tatal = 6090
    data_set_mouthRIO_T1 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/mouth_ROI_T1_56_6090_40_40pixels/'
        + 'mouth_ROI_T1_56_6090_40_40pixels.npy') / 255
    data_set_mouthRIO_T2 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/mouth_ROI_T2_56_6090_40_40pixels/'
        + 'mouth_ROI_T2_56_6090_40_40pixels.npy') / 255
    # data_set_mouthRIO_T1 = data_set_mouthRIO_T1[:, 0:img_num_tatal, :, :, :]
    # data_set_mouthRIO_T2 = data_set_mouthRIO_T2[:, 0:img_num_tatal, :, :, :]
    data_set_mouthRIO_T1 = img_dowm_sample(data_set_mouthRIO_T1)
    data_set_mouthRIO_T2 = img_dowm_sample(data_set_mouthRIO_T2)
    data_set_mouthRIO = np.concatenate((data_set_mouthRIO_T1, data_set_mouthRIO_T2), axis=0)  # (56, 6090, 40, 40, 3)
    data_set_mouthRIO = data_set_mouthRIO.transpose((0, 4, 2, 3, 1))
    print('data_set_mouthRIO_T1.shape = ', data_set_mouthRIO_T1.shape)
    print('data_set_mouthRIO_T2.shape = ', data_set_mouthRIO_T2.shape)
    print('data_set_mouthRIO.shape = ', data_set_mouthRIO.shape)
    del data_set_mouthRIO_T1, data_set_mouthRIO_T2

    # 2. 左眼
    data_set_eyeLeftRIO_T1 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/eyeLeft_ROI_T1_56_6090_40_40pixels/'
        + 'eyeLeft_ROI_T1_56_6090_40_40pixels.npy') / 255
    data_set_eyeLeftRIO_T2 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/eyeLeft_ROI_T2_56_6090_40_40pixels/'
        + 'eyeLeft_ROI_T2_56_6090_40_40pixels.npy') / 255
    # data_set_eyeLeftRIO_T1 = data_set_eyeLeftRIO_T1[:, 0:img_num_tatal, :, :, :]
    # data_set_eyeLeftRIO_T2 = data_set_eyeLeftRIO_T2[:, 0:img_num_tatal, :, :, :]
    data_set_eyeLeftRIO_T1 = img_dowm_sample(data_set_eyeLeftRIO_T1)
    data_set_eyeLeftRIO_T2 = img_dowm_sample(data_set_eyeLeftRIO_T2)
    data_set_eyeLeftRIO = np.concatenate((data_set_eyeLeftRIO_T1, data_set_eyeLeftRIO_T2), axis=0)
    data_set_eyeLeftRIO = data_set_eyeLeftRIO.transpose((0, 4, 2, 3, 1))
    print('data_set_eyeLeftRIO_T1.shape = ', data_set_eyeLeftRIO_T1.shape)
    print('data_set_eyeLeftRIO_T2.shape = ', data_set_eyeLeftRIO_T2.shape)
    print('data_set_eyeLeftRIO.shape = ', data_set_eyeLeftRIO.shape)
    del data_set_eyeLeftRIO_T1, data_set_eyeLeftRIO_T2

    # 3. 右眼
    data_set_eyeRightRIO_T1 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/eyeRight_ROI_T1_56_6090_40_40pixels/'
        + 'eyeRight_ROI_T1_56_6090_40_40pixels.npy') / 255
    data_set_eyeRightRIO_T2 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/eyeRight_ROI_T2_56_6090_40_40pixels/'
        + 'eyeRight_ROI_T2_56_6090_40_40pixels.npy') / 255
    # data_set_eyeRightRIO_T1 = data_set_eyeRightRIO_T1[:, 0:img_num_tatal, :, :, :]
    # data_set_eyeRightRIO_T2 = data_set_eyeRightRIO_T2[:, 0:img_num_tatal, :, :, :]
    data_set_eyeRightRIO_T1 = img_dowm_sample(data_set_eyeRightRIO_T1)
    data_set_eyeRightRIO_T2 = img_dowm_sample(data_set_eyeRightRIO_T2)
    data_set_eyeRightRIO = np.concatenate((data_set_eyeRightRIO_T1, data_set_eyeRightRIO_T2), axis=0)
    data_set_eyeRightRIO = data_set_eyeRightRIO.transpose((0, 4, 2, 3, 1))
    print('data_set_eyeRightRIO_T1.shape = ', data_set_eyeRightRIO_T1.shape)
    print('data_set_eyeRightRIO_T2.shape = ', data_set_eyeRightRIO_T2.shape)
    print('data_set_eyeRightRIO.shape = ', data_set_eyeRightRIO.shape)
    del data_set_eyeRightRIO_T1, data_set_eyeRightRIO_T2

    # ********************* 4. 头部位姿：数据处理************************************************************* #
    # 4. 头部位姿：数据处理
    # data_set_loc_x_T1 = np.load(r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/loc_T1_56_6090_468/'
    #                           +'loc_x_T1_56_6090_468.npy')
    # data_set_loc_y_T1 = np.load(r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/loc_T1_56_6090_468/'
    #                           +'loc_y_T1_56_6090_468.npy')
    # data_set_loc_z_T1 = np.load(r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/loc_T1_56_6090_468/'
    #                           +'loc_z_T1_56_6090_468.npy')
    # data_set_loc_x_T2 = np.load(
    #     r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/loc_T2_56_6090_468/'
    #     + 'loc_x_T2_56_6090_468.npy')
    # data_set_loc_y_T2 = np.load(
    #     r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/loc_T2_56_6090_468/'
    #     + 'loc_y_T2_56_6090_468.npy')
    # data_set_loc_z_T2 = np.load(
    #     r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/loc_T2_56_6090_468/'
    #     + 'loc_z_T2_56_6090_468.npy')
    # loc_arr_T1 = data_pre(data_set_loc_x_T1, data_set_loc_y_T1, data_set_loc_z_T1)
    # loc_arr_T2 = data_pre(data_set_loc_x_T2, data_set_loc_y_T2, data_set_loc_z_T2)
    # save_path_T1 = '/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/loc_T1_56_6090_468/'
    # save_path_T2 = '/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/loc_T2_56_6090_468/'
    # np.save(save_path_T1 + 'loc_distan_T1_56_6090_468' + '.npy',
    #         np.array(loc_arr_T1))
    # np.save(save_path_T2 + 'loc_distan_T2_56_6090_468' + '.npy',
    #         np.array(loc_arr_T2))
    # print('loc_arr_T1.shape = ', loc_arr_T1.shape) # (56, 6090, 468)
    # print('loc_arr_T2.shape = ', loc_arr_T2.shape) # (56, 6090, 468)
    # data_set_loc = np.concatenate((loc_arr_T1, loc_arr_T2), axis=0) #(112, 6090, 468)
    # print('data_set_loc.shape = ', data_set_loc.shape)
    # print('data_set_loc_x_T1.shape = ', data_set_loc_x_T1.shape) #(56, 6090, 468)
    # print('data_set_loc_y_T1.shape = ', data_set_loc_y_T1.shape)
    # print('data_set_loc_z_T1.shape = ', data_set_loc_z_T1.shape)
    #
    # print('data_set_loc_x_T2.shape = ', data_set_loc_x_T2.shape)
    # print('data_set_loc_y_T2.shape = ', data_set_loc_y_T2.shape)
    # print('data_set_loc_z_T2.shape = ', data_set_loc_z_T2.shape)
    # ********************* 4. 头部位姿：数据处理************************************************************* #

    # 4.头部位姿：
    loc_arr_T1 = np.load(
        '/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/loc_T1_56_6090_468/'
        + 'loc_distan_T1_56_6090_468.npy')
    loc_arr_T2 = np.load(
        '/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/loc_T2_56_6090_468/'
        + 'loc_distan_T2_56_6090_468.npy')
    print('loc_arr_T1.shape = ', loc_arr_T1.shape)  # (56, 6090, 468)
    print('loc_arr_T2.shape = ', loc_arr_T2.shape)  # (56, 6090, 468)
    loc = np.concatenate((loc_arr_T1, loc_arr_T2), axis=0)
    print('loc.shape = ', loc.shape)  # (112, 6090, 468)
    del loc_arr_T1, loc_arr_T2
    # 二、 生理数据：额头iPPG
    # 6. forehead iPPG  单个ippg = (3, 6090)
    data_set_ippg_forehead_T1 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/ippg_forehead_T1_56_3_6090/'
        + 'ippg_forehead_T1_56_3_6090.npy') / 255
    data_set_ippg_forehead_T2 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/ippg_forehead_T2_56_3_6090/'
        + 'ippg_forehead_T2_56_3_6090_notNAN.npy') / 255
    # print('data_set_ippg_forehead_T1.shape = ', data_set_ippg_forehead_T1.shape)#(56, 3, 6090)
    # print('data_set_ippg_forehead_T2.shape = ', data_set_ippg_forehead_T2.shape)
    # 原始ippg： ippg_forehead_T1[:,0:3,:], HR_ippg: ippg_forehead_T1[:,3:6,:], RR_ippg: ippg_forehead_T1[:,6:9,:]
    ippg_forehead_T1 = get_time_domain(data_set_ippg_forehead_T1)  # (56, 9, 6090)
    ippg_forehead_T2 = get_time_domain(data_set_ippg_forehead_T2)  # (56, 9, 6090)
    data_set_ippg_forehead = np.concatenate((ippg_forehead_T1, ippg_forehead_T2), axis=0)
    print('data_set_ippg_forehead.shape = ', data_set_ippg_forehead.shape)  # (112, 9, 6090)

    del data_set_ippg_forehead_T1, data_set_ippg_forehead_T2, ippg_forehead_T1, ippg_forehead_T2

    # # 6. nose iPPG
    data_set_ippg_nose_T1 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/ippg_nose_T1_56_3_6090/'
        + 'ippg_nose_T1_56_3_6090.npy') / 255
    data_set_ippg_nose_T2 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/ippg_nose_T2_56_3_6090/'
        + 'ippg_nose_T2_56_3_6090.npy') / 255
    print('data_set_ippg_nose_T1.shape = ', data_set_ippg_nose_T1.shape)  # (56, 3, 6090)
    print('data_set_ippg_nose_T2.shape = ', data_set_ippg_nose_T2.shape)  # (56, 3, 6090)
    ippg_nose_T1 = get_time_domain(data_set_ippg_nose_T1)  # (56, 9, 6090)
    ippg_nose_T2 = get_time_domain(data_set_ippg_nose_T2)  # (56, 9, 6090)
    data_set_ippg_nose = np.concatenate((ippg_nose_T1, ippg_nose_T2), axis=0)
    print('data_set_ippg_nose.shape = ', data_set_ippg_nose.shape)  # (112, 9, 6090)
    del data_set_ippg_nose_T1, data_set_ippg_nose_T2, ippg_nose_T1, ippg_nose_T2

    # ippg
    iPPG = np.concatenate((data_set_ippg_forehead, data_set_ippg_nose), axis=1)  # (112, 18, 6090)
    iPPG = iPPG.transpose((0, 2, 1))  # (112, 6090, 18)
    print('iPPG.shape = ', iPPG.shape)

    # 划分数据集
    # 标签
    y_labels_train, y_labels_val = train_test_split(label, test_size=test_size,
                                                    random_state=random_state)
    y_labels_train = np.array(y_labels_train)
    y_labels_val = np.array(y_labels_val)
    y_labels_train = binary_onehot(y_labels_train)
    y_labels_val = binary_onehot(y_labels_val)
    # 嘴巴
    data_set_mouthRIO_train, data_set_mouthRIO_test = train_test_split(data_set_mouthRIO, test_size=test_size,
                                                                       random_state=random_state)
    # 左眼
    data_set_eyeLeftRIO_train, data_set_eyeLeftRIO_test = train_test_split(data_set_eyeLeftRIO,
                                                                           test_size=test_size,
                                                                           random_state=random_state)
    # 右眼
    data_set_eyeRightRIO_train, data_set_eyeRightRIO_test = train_test_split(data_set_eyeRightRIO, test_size=test_size,
                                                                             random_state=random_state)
    # 头部位姿
    data_set_location_train, data_set_location_test = train_test_split(loc, test_size=test_size,
                                                                       random_state=random_state)

    # ippg
    data_set_iPPG_train, data_set_iPPG_test = train_test_split(iPPG, test_size=test_size,
                                                               random_state=random_state)

    # # fohead ippg
    # data_set_fohead_iPPG_train, data_set_fohead_iPPG_test = train_test_split(data_set_ippg_forehead, test_size=test_size,
    #                                                            random_state=random_state)
    #
    # # nose ippg
    # data_set_nose_iPPG_train, data_set_nose_iPPG_test = train_test_split(data_set_ippg_nose, test_size=test_size,
    #                                                            random_state=random_state)

    del data_set_mouthRIO, data_set_eyeLeftRIO, data_set_eyeRightRIO, loc, iPPG, data_set_ippg_forehead, data_set_ippg_nose

    # initiates model and loss
    no_classes = 2
    model = ConvNet3D_one(no_classes).to(device)
    # model_ConvNet3D_two = ConvNet3D_two(no_classes).to(device)

    # criterion = nn.CrossEntropyLoss()  # alternatively MSE if regression or PSNR/PSD or pearson correlation
    criterion = nn.BCELoss()  # https://blog.csdn.net/weixin_40522801/article/details/106616564
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ###
    total_step = len(y_labels_train) // batch_size
    curr_lr = learning_rate
    training_proc_avg = []
    test_proc_avg = []
    for epoch in range(no_epochs):

        ## 训练
        current_losses = []
        total_loss = 0.0
        accuracy = 0
        y_true = []
        y_pred = []
        y_pred_probability = []
        for i in range(total_step):
            if i < total_step - 1:
                # print('#################')
                # print('y_labels_train.shape = ',y_labels_train.shape)
                labels = torch.from_numpy(y_labels_train[i * batch_size:(i + 1) * batch_size]).to(device).float()
                input_shape_mouth= torch.from_numpy(data_set_mouthRIO_train[i * batch_size:(i + 1) * batch_size]).to(device).float()
                input_shape_eyeLeft = torch.from_numpy(data_set_eyeLeftRIO_train[i * batch_size:(i + 1) * batch_size]).to(device).float()
                input_shape_eyeRight = torch.from_numpy(data_set_eyeRightRIO_train[i * batch_size:(i + 1) * batch_size]).to(device).float()
                input_shape_xyz = torch.from_numpy(data_set_location_train[i * batch_size:(i + 1) * batch_size]).to(device).float()
                # input_shape_iPPG = torch.from_numpy(data_set_iPPG_train[i * batch_size:(i + 1) * batch_size]).to(device).float()
            else:
                labels = torch.from_numpy(y_labels_train[i * batch_size:-1]).to(device).float()
                input_shape_mouth= torch.from_numpy(data_set_mouthRIO_train[i * batch_size:-1]).to(device).float()
                input_shape_eyeLeft = torch.from_numpy(data_set_eyeLeftRIO_train[i * batch_size:-1]).to(device).float()
                input_shape_eyeRight = torch.from_numpy(data_set_eyeRightRIO_train[i * batch_size:-1]).to(device).float()
                input_shape_xyz = torch.from_numpy(data_set_location_train[i * batch_size:-1]).to(device).float()
                # input_shape_iPPG = torch.from_numpy(data_set_iPPG_train[i * batch_size:-1]).to(device).float()


            # Forward pass
            optimizer.zero_grad()
            model.train()
            # print('input_shape_mouth.shape', input_shape_mouth.shape)
            # print('input_shape_eyeLeft.shape', input_shape_eyeLeft.shape)
            # print('input_shape_eyeRight.shape', input_shape_eyeRight.shape)
            # print('input_shape_eTou.shape', input_shape_eTou.shape)
            outputs = model(
                          input_shape_mouth,
                          input_shape_eyeLeft, input_shape_eyeRight,
                        input_shape_xyz,
                        # input_shape_iPPG,
                        )
            # print('######')
            # print('outputs = ', outputs)
            # print('labels = ', labels )
            # print('######')
            loss = criterion(outputs, labels)
            # loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i + 1) % 32 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}'
                      .format(epoch + 1, no_epochs, i + 1, total_step, loss.item()))
                current_losses.append(loss.item())  # appends the current value of the loss into a list

                y_pred, y_true, y_pred_probability = validate(device, batch_size, model,criterion,
                         y_labels_val,
                         data_set_mouthRIO_test,
                         data_set_eyeLeftRIO_test, data_set_eyeRightRIO_test,
                          data_set_location_test,
                          # data_set_iPPG_test,
                         training_proc_avg, test_proc_avg, last=False)

    # 模型性能相关指标  https://zhuanlan.zhihu.com/p/110015537       https://www.plob.org/article/12476.html
    y_test = y_true
    preds = y_pred
    test_predict = y_pred_probability
    acc = accuracy_score(y_test, preds)
    roc_auc = roc_auc_score(y_test,
                            test_predict)  # https://blog.csdn.net/u013385925/article/details/80385873    https://www.plob.org/article/12476.html
    precision = precision_score(y_test, preds)
    # recall = recall_score(y_test, preds,average='macro')
    sensitivity = sensitivity_score(y_test, preds)
    # f1 = f1_score(y_test, preds, average='weighted')
    gmean = geometric_mean_score(y_test, preds)  # 几何平均
    mat = confusion_matrix(y_test, preds)
    fpr, tpr, threshold = roc_curve(y_test, test_predict)  ###计算真正率和假正率
    ## # 计算准确率
    y_true = y_test
    y_pred = preds
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    # acc1 = accuracy_score(y_true, y_pred)
    # 计算精确率
    precision1 = metrics.precision_score(y_true, y_pred)
    # 计算召回率
    recall1 = metrics.recall_score(y_true, y_pred)
    # 计算f得分
    fscore1 = metrics.f1_score(y_true, y_pred)
    # 计算敏感性
    sensitivity1 = sensitivity_score(y_true, y_pred)
    # print('acc1 = ', acc1)
    print('precision1 = ', precision1)
    print('recall1 = ', recall1)
    print('fscore1 = ', fscore1)
    print('sensitivity1 = ', sensitivity1)
    print('specificity = ', specificity)
    #################################
    print('fpr:', fpr)
    print('tpr:', tpr)
    print('threshold:', threshold)
    print("Accuracy: ", acc)
    print("ROC_Auc: ", roc_auc)
    print("Precision:", precision)
    # print("Recall:", recall)
    print("Sensitivity:", sensitivity)
    # print("F1 score: ", f1)
    print('G-mean:', gmean)
    print("Confusion matrix: \n", mat)
    print('Overall report: \n', classification_report(y_test, preds))
    # save result
    save_name = '_' + 'epochs' + str(no_epochs) + '_' + 'mouth' + '_' + 'eyeLeft' + '_' + 'eyeRight' + '_' + 'location'
    # + '_' + 'mouth' \
    # + '_' + 'eyeLeft' + '_' + 'eyeRight' \
    # + '_' + 'location'\
    # + '_' + 'forehead'\
    # + '_' + 'iPPG' # 额头提取iPPG
    data_result = []
    data_result.append([acc, roc_auc, precision, recall1, sensitivity, fscore1, specificity])
    # np.savetxt('resnet_3D_fpr' + save_name + '.txt', fpr)
    # np.savetxt('resnet_3D_tpr' + save_name + '.txt', tpr)
    # np.savetxt('3DCNN_data_result' + save_name + '.txt', data_result)
    # np.savetxt('3DCNN_y_test' + save_name + '.txt', y_true)
    # np.savetxt('3DCNN_y_pred_probability' + save_name + '.txt', y_pred_probability)
    # np.savetxt('3DCNN_y_pred' + save_name + '.txt', y_pred)

    np.savetxt('./3DCNN_data_result_40_40pixel/3DCNN_data_result' + save_name + '.txt', np.array(data_result))
    np.savetxt('./3DCNN_data_result_40_40pixel/3DCNN_y_test' + save_name + '.txt', y_true)
    np.savetxt('./3DCNN_data_result_40_40pixel/3DCNN_y_pred_probability' + save_name + '.txt', y_pred_probability)
    np.savetxt('./3DCNN_data_result_40_40pixel/3DCNN_y_pred' + save_name + '.txt', y_pred)
    # plt.figure()
    import matplotlib.pyplot as plt
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic of TextCNN')
    plt.legend(loc="lower right")
    # plt.savefig('./imgs/FastText.png')
    # plt.show()
    # 保存模型
    # model.save('3DCNN_mod4_0_1_anxiety_screeen' + save_name + '.h5')
    torch.save(model, './3DCNN_data_result_40_40pixel/3DCNN_mod4_0_1_anxiety_screeen' + save_name + '.pkl')

if __name__ == '__main__':
    train()