
from torch.autograd import Variable
from collections import OrderedDict
import torch.nn.functional as F
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
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
print('torch.cuda.device_count()=', torch.cuda.device_count())
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def validate(device, batch_size, model,criterion,
             y_labels_val,
             data_set_mouthRIO_test,
             data_set_eyeLeftRIO_test, data_set_eyeRightRIO_test,
            data_set_xyz_test,
             # data_set_eTouRIO_test,
             training_proc_avg, test_proc_avg, last=False):
    # Test the model (validation set)
    gen_signals_val = data_set_mouthRIO_test
    y_labels_train  = y_labels_val
    ##
    model = model.to(device)
    model.eval()
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
                target = torch.from_numpy(y_labels_train[i * batch_size:(i + 1) * batch_size]).to(device)
                input_shape_mouth = torch.from_numpy(data_set_mouthRIO_test[i * batch_size:(i + 1) * batch_size]).to(device).float()
                input_shape_eyeLeft = torch.from_numpy(data_set_eyeLeftRIO_test[i * batch_size:(i + 1) * batch_size]).to(device).float()
                input_shape_eyeRight = torch.from_numpy(data_set_eyeRightRIO_test[i * batch_size:(i + 1) * batch_size]).to(device).float()
                input_shape_xyz = torch.from_numpy(data_set_xyz_test[i * batch_size:(i + 1) * batch_size]).to(device).float()
                # input_shape_eTou = torch.from_numpy(data_set_eTouRIO_test[i * batch_size:(i + 1) * batch_size]).to(device).float()
            else:
                target = torch.from_numpy(y_labels_train[i * batch_size:-1]).to(device).to(device)
                input_shape_mouth = torch.from_numpy(data_set_mouthRIO_test[i * batch_size:-1]).to(device).float()
                input_shape_eyeLeft = torch.from_numpy(data_set_eyeLeftRIO_test[i * batch_size:-1]).to(device).float()
                input_shape_eyeRight = torch.from_numpy(data_set_eyeRightRIO_test[i * batch_size:-1]).to(device).float()
                input_shape_xyz = torch.from_numpy(data_set_xyz_test[i * batch_size:-1]).to(device).float()
                # input_shape_eTou = torch.from_numpy(data_set_eTouRIO_test[i * batch_size:-1]).to(device).float()

                # Forward pass
            out = model(
                        input_shape_mouth,
                        input_shape_eyeLeft, input_shape_eyeRight,
                        input_shape_xyz,
                        # input_shape_eTou,
                        )
            # print('out = ', out)
            # print('target = ',target)
            # loss = criterion(out, target)
            loss = F.cross_entropy(out, target)
            total_loss += loss.item()
            # accuracy += (torch.argmax(out, dim=1) == target).sum().item()
            # y_true.extend(torch.argmax(target, dim=1).cpu().numpy())
            # y_pred.extend(torch.argmax(out, dim=1).cpu().numpy())
            # y_pred_probability_arr.append(out.cpu().numpy())
            # print('out = ', out)
            # print('target = ', target)
            # print('torch.argmax(out, dim=1)  =', torch.argmax(out, dim=1))

            accuracy += (torch.argmax(out, dim=1) == target).sum().item()
            y_true.extend(target.cpu().numpy())
            y_pred.extend(torch.argmax(out, dim=1).cpu().numpy())
            y_pred_probability.extend(F.sigmoid(out).cpu().detach().numpy())  # 计算softmax，即属于各类的概率


        

    y_pred_probability = np.array(y_pred_probability)
    # print('y_pred  = ', y_pred)
    # print('y_true  = ', y_true)
    # print('y_pred_probability  = ', y_pred_probability)
    # print('##############')
    y_pred_probability = y_pred_probability[:, 1]
    # print('y_pred_probability  = ', y_pred_probability)

    acc = accuracy_score(y_true, y_pred)
    print('acc = ', acc)
    # print('y_true = ', y_true)
    # print('y_pred_probability = ', y_pred_probability)
    # print('np.array(y_pred).shape = ', np.array(y_pred).shape)
    # print('np.array(y_true).shape = ', np.array(y_true).shape)
    # print('np.array(y_pred_probability).shape = ', np.array(y_pred_probability).shape)
    # print('np.array(y_labels_val).shape = ', np.array(y_labels_val).shape)

    return y_pred, y_true, y_pred_probability, acc


import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=2):
        super().__init__()

        self.in_channels = 40

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.n_units = 128
        self.picture_input = 2030 * 2048
        self.fc_picture = nn.Sequential(
            nn.Linear(self.picture_input, self.n_units),
            # nn.Linear(128, num_classes),
            # nn.Sigmoid(),  # 和nn.BCELoss()配合使用，详见https://blog.csdn.net/weixin_40522801/article/details/106616564
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(153 * 3 * 14, self.n_units),
        )

        self.fc_iPPG = nn.Sequential(
            nn.Linear(153 * 3 * 14, self.n_units),
        )
        ###  ippg
        self.n_inputs_iPPG = 18
        self.rnn_iPPG = nn.LSTM(
            input_size=self.n_inputs_iPPG,
            hidden_size=self.n_units,
            num_layers=1,
            batch_first=True,
        )
        ###
        ## LSTM
        # self.n_units = 128
        self.n_inputs = 468  # 3DCNN输出的特征大小.shape[2] = 200
        self.rnn = nn.LSTM(
            input_size=self.n_inputs,
            hidden_size=self.n_units,
            num_layers=1,
            batch_first=True,
        )
        ###############################
        num_features = 4 # 根据输入特征的个数进行修改
        # fc_sum_inputnum = 100*2048
        self.fc_sum = nn.Sequential(
            # nn.Linear(num_features * fc_sum_inputnum, 128),
            nn.Linear(num_features*128, num_classes),
            # nn.Sigmoid(),  # 和nn.BCELoss()配合使用，详见https://blog.csdn.net/weixin_40522801/article/details/106616564
        )


    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)


    def feature_extract(self, input_shape_mouth):
        features_arr = []
        for i in range(input_shape_mouth.shape[0]):
            # for j in range(input_shape_mouth.shape[0]):
            output = input_shape_mouth[i]
            # print('output.shape=', output.shape)
            output = self.conv2_x(output)
            output = self.conv3_x(output)
            output = self.conv4_x(output)
            output = self.conv5_x(output)
            output = self.avg_pool(output)
            output = output.view(output.size(0), -1)
            output = torch.reshape(output, (-1,))
            # print('out2.shape = ', out.shape)
            # out = out.cpu().detach().numpy()
            output = output.cpu().detach().numpy()
            features_arr.append(output)
        features_arr = np.array(features_arr)
        # print('type(features_arr) = ', type(features_arr))
        # print('features_arr.shape = ', features_arr.shape)
        # tensor.reshape
        features_arr = self.fc_picture(torch.from_numpy(features_arr).to(device))
        return features_arr

    def feature_extract_iPPG(self, x):
        # print('x.shape= ', x.shape)
        x = x.to(device).float()
        r_out, (h_n, h_c) = self.rnn_iPPG(x, None)
        # out = self.out(r_out[:, -1, :])
        out = r_out[:, -1, :]
        return out

    def feature_extract_LSTM(self, x):
        # print('x.shape= ', x.shape)
        x = x.to(device).float()
        r_out, (h_n, h_c) = self.rnn(x, None)
        # out = self.out(r_out[:, -1, :])
        out = r_out[:, -1, :]
        return out

    def forward(self,
                input_shape_mouth,
                input_shape_eyeLeft, input_shape_eyeRight,
                input_shape_xyz,
                # input_shape_eTou,
                ):

        # print('input_shape_mouth.shape = ', input_shape_mouth.shape)
        # print('input_shape_mouth.shape=', input_shape_mouth.shape)
        features_mouth = self.feature_extract(input_shape_mouth)
        features_eyeLeft = self.feature_extract(input_shape_eyeLeft)
        features_eyeRight = self.feature_extract(input_shape_eyeRight)
        features_xyz = self.feature_extract_LSTM(input_shape_xyz)
        # features_eTou = self.feature_extract(input_shape_eTou)
        # print('features_xyz.shape = ', features_xyz.shape)
        # print('features_eyeRight.shape = ', features_eyeRight.shape)

        out = torch.cat((
            features_mouth,
            features_eyeLeft, features_eyeRight,
            features_xyz,
            # features_eTou,
        ), 1)  # 在 1 维(横向)进行拼接
        # print('out.shape = ', out.shape)

        out = self.fc_sum(out)
        # x = input_shape_mouth
        # print
        # output = self.conv1(x)
        # output = self.conv2_x(output)
        # output = self.conv3_x(output)
        # output = self.conv4_x(output)
        # output = self.conv5_x(output)
        # output = self.avg_pool(output)
        # output = output.view(output.size(0), -1)
        # output = self.fc(output)

        return out

def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])

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

def binary_onehot(label_set):
    # print('label_set = ', label_set)
    label_arr = []
    for i in range(len(label_set)):
        if label_set[i] != 0:
            # label_set[i] = 1
            label_arr.append([0, 1])
        else:
            # label_set[i] = 0
            label_arr.append([1, 0])
    return np.array(label_arr)


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
    img_num_per_second = 3
    data_arr = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if j % img_num_per_second == 0:
                data_arr.append(data[i, j])
    data_arr = np.reshape(data_arr, (
    data.shape[0], int(data.shape[1] / img_num_per_second), data.shape[2], data.shape[3], data.shape[4]))
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
    data_set_mouthRIO_T1 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/mouth_ROI_T1_56_6090_40_40pixels/'
        + 'mouth_ROI_T1_56_6090_40_40pixels.npy')/ 255
    data_set_mouthRIO_T2 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/mouth_ROI_T2_56_6090_40_40pixels/'
        + 'mouth_ROI_T2_56_6090_40_40pixels.npy')/ 255
    data_set_mouthRIO_T1 = img_dowm_sample(data_set_mouthRIO_T1)
    data_set_mouthRIO_T2 = img_dowm_sample(data_set_mouthRIO_T2)
    data_set_mouthRIO = np.concatenate((data_set_mouthRIO_T1, data_set_mouthRIO_T2), axis=0)  # (56, 6090, 40, 40, 3)
    # data_set_mouthRIO = data_set_mouthRIO.transpose((0, 1, 4, 2, 3))
    print('data_set_mouthRIO_T1.shape = ', data_set_mouthRIO_T1.shape)
    print('data_set_mouthRIO_T2.shape = ', data_set_mouthRIO_T2.shape)
    print('data_set_mouthRIO.shape = ', data_set_mouthRIO.shape)
    del data_set_mouthRIO_T1, data_set_mouthRIO_T2

    # 2. 左眼
    data_set_eyeLeftRIO_T1 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/eyeLeft_ROI_T1_56_6090_40_40pixels/'
        + 'eyeLeft_ROI_T1_56_6090_40_40pixels.npy')/ 255
    data_set_eyeLeftRIO_T2 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/eyeLeft_ROI_T2_56_6090_40_40pixels/'
        + 'eyeLeft_ROI_T2_56_6090_40_40pixels.npy')/ 255
    data_set_eyeLeftRIO_T1 = img_dowm_sample(data_set_eyeLeftRIO_T1)
    data_set_eyeLeftRIO_T2 = img_dowm_sample(data_set_eyeLeftRIO_T2)
    data_set_eyeLeftRIO = np.concatenate((data_set_eyeLeftRIO_T1, data_set_eyeLeftRIO_T2), axis=0)
    # data_set_eyeLeftRIO = data_set_eyeLeftRIO.transpose((0, 1, 4, 2, 3))
    print('data_set_eyeLeftRIO_T1.shape = ', data_set_eyeLeftRIO_T1.shape)
    print('data_set_eyeLeftRIO_T2.shape = ', data_set_eyeLeftRIO_T2.shape)
    print('data_set_eyeLeftRIO.shape = ', data_set_eyeLeftRIO.shape)
    del data_set_eyeLeftRIO_T1, data_set_eyeLeftRIO_T2

    # 3. 右眼
    data_set_eyeRightRIO_T1 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/eyeRight_ROI_T1_56_6090_40_40pixels/'
        + 'eyeRight_ROI_T1_56_6090_40_40pixels.npy')/ 255
    data_set_eyeRightRIO_T2 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/eyeRight_ROI_T2_56_6090_40_40pixels/'
        + 'eyeRight_ROI_T2_56_6090_40_40pixels.npy')/ 255
    data_set_eyeRightRIO_T1 = img_dowm_sample(data_set_eyeRightRIO_T1)
    data_set_eyeRightRIO_T2 = img_dowm_sample(data_set_eyeRightRIO_T2)
    data_set_eyeRightRIO = np.concatenate((data_set_eyeRightRIO_T1, data_set_eyeRightRIO_T2), axis=0)
    # data_set_eyeRightRIO = data_set_eyeRightRIO.transpose((0, 1, 4, 2, 3))
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
    location = np.concatenate((loc_arr_T1, loc_arr_T2), axis=0)
    print('location.shape = ', location.shape)  # (112, 6090, 468)
    del loc_arr_T1, loc_arr_T2
    # 二、 生理数据：额头iPPG
    # 6. forehead iPPG  单个ippg = (3, 6090)
    data_set_ippg_forehead_T1 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/ippg_forehead_T1_56_3_6090/'
        + 'ippg_forehead_T1_56_3_6090.npy')/ 255
    data_set_ippg_forehead_T2 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/ippg_forehead_T2_56_3_6090/'
        + 'ippg_forehead_T2_56_3_6090.npy')/ 255
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
        + 'ippg_nose_T1_56_3_6090.npy')/ 255
    data_set_ippg_nose_T2 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/ippg_nose_T2_56_3_6090/'
        + 'ippg_nose_T2_56_3_6090.npy')/ 255
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
    # y_labels_train = binary_onehot(y_labels_train)
    # y_labels_val = binary_onehot(y_labels_val)
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
    data_set_location_train, data_set_location_test = train_test_split(location, test_size=test_size,
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

    del data_set_mouthRIO, data_set_eyeLeftRIO, data_set_eyeRightRIO, location, iPPG, data_set_ippg_forehead, data_set_ippg_nose
    # initiates model and loss
    model = ResNet(BottleNeck, [3, 4, 6, 3]).to(device)
    criterion = nn.CrossEntropyLoss()  # alternatively MSE if regression or PSNR/PSD or pearson correlation
    # criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ###
    print('len(y_labels_train)  = ', len(y_labels_train))
    total_step = len(y_labels_train) // batch_size
    curr_lr = learning_rate
    training_proc_avg = []
    test_proc_avg = []
    temp_acc = 0
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
                labels = torch.from_numpy(y_labels_train[i * batch_size:(i + 1) * batch_size]).to(device)
                input_shape_mouth = torch.from_numpy(data_set_mouthRIO_train[i * batch_size:(i + 1) * batch_size, :]).to(device).float()
                input_shape_eyeLeft = torch.from_numpy(data_set_eyeLeftRIO_train[i * batch_size:(i + 1) * batch_size]).to(device).float()
                input_shape_eyeRight = torch.from_numpy(data_set_eyeRightRIO_train[i * batch_size:(i + 1) * batch_size]).to(device).float()
                input_shape_location = torch.from_numpy(data_set_location_train[i * batch_size:(i + 1) * batch_size]).to(device).float()
                # input_shape_forehead = torch.from_numpy(data_set_forehead_train[i * batch_size:(i + 1) * batch_size, :]).to(device).float()
            else:
                labels = torch.from_numpy(y_labels_train[i * batch_size:-1]).to(device)
                input_shape_mouth = torch.from_numpy(data_set_mouthRIO_train[i * batch_size:-1]).to(device).float()
                input_shape_eyeLeft = torch.from_numpy(data_set_eyeLeftRIO_train[i * batch_size:-1]).to(device).float()
                input_shape_eyeRight = torch.from_numpy(data_set_eyeRightRIO_train[i * batch_size:-1]).to(device).float()
                input_shape_location = torch.from_numpy(data_set_location_train[i * batch_size:-1]).to(device).float()
                # input_shape_forehead = torch.from_numpy(data_set_forehead_train[i * batch_size:-1]).to(device).float()
            # Forward pass
            optimizer.zero_grad()
            model.train()
            with torch.no_grad():
                outputs = model(
                                input_shape_mouth,
                                input_shape_eyeLeft, input_shape_eyeRight,
                                input_shape_location,
                                # input_shape_forehead,
                                )
                # loss = criterion(outputs, labels)
                loss = F.cross_entropy(outputs, labels)
                loss = loss.requires_grad_()
                loss.backward()
                optimizer.step()
                if (i + 1) % 34 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}'
                          .format(epoch + 1, no_epochs, i + 1, total_step, loss.item()))
                    current_losses.append(loss.item())  # appends the current value of the loss into a list

                    y_pred, y_true, y_pred_probability,acc = validate(device, batch_size, model, criterion,
                                                                  y_labels_val,
                                                                  data_set_mouthRIO_test,
                                                                  data_set_eyeLeftRIO_test, data_set_eyeRightRIO_test,
                                                                  data_set_location_test,
                                                                  # data_set_forehead_test,
                                                                  training_proc_avg, test_proc_avg, last=False)
                    if acc > temp_acc:
                        temp_acc = acc
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
                        np.savetxt('./data_result_resnet_40_40pixel/resnet_data_result' + save_name + '.txt',data_result)
                        np.savetxt('./data_result_resnet_40_40pixel/resnet_y_test' + save_name + '.txt', y_true)
                        np.savetxt('./data_result_resnet_40_40pixel/resnet_y_pred_probability--' + save_name + '.txt', y_pred_probability)
                        np.savetxt('./data_result_resnet_40_40pixel/resnet_y_pred' + save_name + '.txt', y_pred)
                        # 保存模型
                        torch.save(model, './data_result_resnet_40_40pixel/resnet_mod4_0_1_anxiety_screeen' + save_name + '.pkl')


if __name__ == "__main__":
    # model = get_model(num_classes=600, sample_size=112, width_mult=1.)
    # model = model.cuda()
    # model = nn.DataParallel(model, device_ids=None)
    # print(model)
    #
    # input_var = Variable(torch.randn(8, 3, 16, 112, 112))
    # output = model(input_var)
    # print(output.shape)
    train()