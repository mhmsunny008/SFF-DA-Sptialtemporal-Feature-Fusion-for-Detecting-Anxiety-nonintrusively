import torch
from torch import nn
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
# https://github.com/developer0hye/SKNet-PyTorch
#from thop import profile
#from thop import clever_format
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

def validate(device, batch_size, model,
             y_labels_val,
             data_set_mouthRIO_test,
             data_set_eyeLeftRIO_test, data_set_eyeRightRIO_test,
             data_set_xyz_test,
             data_set_iPPG_test,
             training_proc_avg, test_proc_avg, last=False):
    # Test the model (validation set)
    model.eval()
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
                target = torch.from_numpy(y_labels_train[i * batch_size:(i + 1) * batch_size]).to(device)
                input_shape_mouth = torch.from_numpy(data_set_mouthRIO_test[i * batch_size:(i + 1) * batch_size]).to(device).float()
                input_shape_eyeLeft = torch.from_numpy(data_set_eyeLeftRIO_test[i * batch_size:(i + 1) * batch_size]).to(device).float()
                input_shape_eyeRight = torch.from_numpy(data_set_eyeRightRIO_test[i * batch_size:(i + 1) * batch_size]).to(device).float()
                input_shape_xyz = torch.from_numpy(data_set_xyz_test[i * batch_size:(i + 1) * batch_size]).to(device).float()
                input_shape_iPPG = torch.from_numpy(data_set_iPPG_test[i * batch_size:(i + 1) * batch_size]).to(device).float()

                # input_shape_eTou = torch.from_numpy(data_set_eTouRIO_test[i * batch_size:(i + 1) * batch_size]).to(device).float()
            else:
                target = torch.from_numpy(y_labels_train[i * batch_size:-1]).to(device)
                input_shape_mouth = torch.from_numpy(data_set_mouthRIO_test[i * batch_size:-1]).to(device).float()
                input_shape_eyeLeft = torch.from_numpy(data_set_eyeLeftRIO_test[i * batch_size:-1]).to(device).float()
                input_shape_eyeRight = torch.from_numpy(data_set_eyeRightRIO_test[i * batch_size:-1]).to(device).float()
                input_shape_xyz = torch.from_numpy(data_set_xyz_test[i * batch_size:-1]).to(device).float()
                input_shape_iPPG = torch.from_numpy(data_set_iPPG_test[i * batch_size:-1]).to(device).float()
                # input_shape_eTou = torch.from_numpy(data_set_eTouRIO_test[i * batch_size:-1]).to(device).float()
            # Forward pass
            out = model(
                input_shape_mouth,
                input_shape_eyeLeft, input_shape_eyeRight,
                input_shape_xyz,
                input_shape_iPPG,
                # input_shape_eTou,
            )

            loss = F.cross_entropy(out, target)
            total_loss += loss.item()
            accuracy += (torch.argmax(out, dim=1) == target).sum().item()
            y_true.extend(target.cpu().numpy())
            y_pred.extend(torch.argmax(out, dim=1).cpu().numpy())
            y_pred_probability.extend(F.sigmoid(out).cpu().detach().numpy())  # 计算softmax，即属于各类的概率

    y_pred_probability = np.array(y_pred_probability)
    y_pred_probability = y_pred_probability[:, 1]
    acc = accuracy_score(y_true, y_pred)
    print('acc = ', acc)
    return y_pred, y_true, y_pred_probability, acc

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
class SKConv(nn.Module):
    def __init__(self, features, M=2, G=32, r=16, stride=1 ,L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1+i, dilation=1+i, groups=G, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=False))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                 nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        
        feats = [conv(x) for conv in self.convs]      
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])
        
        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(feats*attention_vectors, dim=1)
        
        return feats_V


class SKUnit(nn.Module):
    def __init__(self, in_features, mid_features, out_features, M=2, G=32, r=16, stride=1, L=32):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True)
            )
        
        self.conv2_sk = SKConv(mid_features, M=M, G=G, r=r, stride=stride, L=L)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_features, out_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_features)
            )
        

        if in_features == out_features: # when dim not change, input_features could be added diectly to out
            self.shortcut = nn.Sequential()
        else: # when dim not change, input_features should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_features)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.conv2_sk(out)
        out = self.conv3(out)
        
        return self.relu(out + self.shortcut(residual))

class SKNet(nn.Module):
    def __init__(self, class_num, nums_block_list = [3, 4, 6, 3], strides_list = [1, 2, 2, 2]):
        super(SKNet, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.num_classes = class_num
        self.maxpool = nn.MaxPool2d(3,2,1)
        
        self.stage_1 = self._make_layer(64, 128, 256, nums_block=nums_block_list[0], stride=strides_list[0])
        self.stage_2 = self._make_layer(256, 256, 512, nums_block=nums_block_list[1], stride=strides_list[1])
        self.stage_3 = self._make_layer(512, 512, 1024, nums_block=nums_block_list[2], stride=strides_list[2])
        self.stage_4 = self._make_layer(1024, 1024, 2048, nums_block=nums_block_list[3], stride=strides_list[3])
     
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        ## LSTM
        self.n_units = 64
        self.n_inputs = 468  # 3DCNN输出的特征大小.shape[2] = 200
        self.rnn = nn.LSTM(
            input_size=self.n_inputs,
            hidden_size=self.n_units,
            num_layers=1,
            batch_first=True,
        )
        ###  ippg
        self.n_inputs_iPPG = 18
        self.rnn_iPPG = nn.LSTM(
            input_size=self.n_inputs_iPPG,
            hidden_size=self.n_units,
            num_layers=1,
            batch_first=True,
        )
        # self.classifier = nn.Linear(2048, class_num)
        input_sknet = 2048
        output_sknet = 10
        self.out_sknet = nn.Linear(input_sknet, output_sknet)
        input_sknet_fc = 2030 * 10
        self.out_sknet_fc = nn.Linear(input_sknet_fc, self.n_units)
        ##
        num_features = 5  # 根据输入特征的个数进行修改
        self.fc_sum = nn.Sequential(
            nn.Linear(self.n_units * num_features, self.num_classes),
            nn.Sigmoid(),  # 和nn.BCELoss()配合使用，详见https://blog.csdn.net/weixin_40522801/article/details/106616564
        )

    def _make_layer(self, in_feats, mid_feats, out_feats, nums_block, stride=1):
        layers=[SKUnit(in_feats, mid_feats, out_feats, stride=stride)]
        for _ in range(1,nums_block):
            layers.append(SKUnit(out_feats, mid_feats, out_feats))
        return nn.Sequential(*layers)

    def feature_sknet(self, x):  # x = torch.rand(batch, C W, H)
        # print('x.shape = ', x.shape)
        x = x.to(device).float()
        out_arr = []
        for i in range(x.shape[0]):
            fea = self.basic_conv(x[i])
            fea = self.maxpool(fea)
            fea = self.stage_1(fea)
            fea = self.stage_2(fea)
            fea = self.stage_3(fea)
            fea = self.stage_4(fea)
            fea = self.gap(fea)
            fea = torch.squeeze(fea)
            fea = self.out_sknet(fea)
            out_arr.append(fea)
        # print('fea.shape = ', fea.shape)# [2030, 10]
        out_arr = torch.stack(out_arr)
        # print('out_arr.shape = ', out_arr.shape)#[2, 2030, 10]
        out_arr = out_arr.reshape((out_arr.shape[0], out_arr.shape[1] * out_arr.shape[2]))  # [2, 2030*10]
        # print('#########out_arr.shape = ', out_arr.shape)  # # [2, 64]
        out_arr = self.out_sknet_fc(out_arr)# [2, 64]
        # print('*************out_arr.shape = ', out_arr.shape) # # [2, 64]
        return out_arr

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
    def forward(self,
              input_shape_mouth,
              input_shape_eyeLeft, input_shape_eyeRight,
                input_shape_location,
                input_shape_iPPG,
                ):
        features_mouth = self.feature_sknet(input_shape_mouth)
        features_eyeLeft = self.feature_sknet(input_shape_eyeLeft)
        features_eyeRight = self.feature_sknet(input_shape_eyeRight)
        features_xyz = self.feature_extract_LSTM(input_shape_location)
        features_iPPG = self.feature_extract_iPPG(input_shape_iPPG)

        # 特征串联
        out = torch.cat((
            features_mouth,
            features_eyeLeft, features_eyeRight,
            features_xyz,
            features_iPPG,
            # features_eTou,
        ), 1)  # 在 1 维(横向)进行拼接
        out = self.fc_sum(out)
        return out


def SKNet26(nums_class=2):
    return SKNet(nums_class, [2, 2, 2, 2])
def SKNet50(nums_class=2):
    return SKNet(nums_class, [3, 4, 6, 3])
def SKNet101(nums_class=2):
    return SKNet(nums_class, [3, 4, 23, 3])
def train():
    # train constants
    no_epochs = 50  # 5000 originally
    # no_videos_by_class = 200
    batch_size = 2
    learning_rate = 1e-4
    no_classes = 2
    # classes = 1
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
        + 'mouth_ROI_T1_56_6090_40_40pixels.npy') / 255
    data_set_mouthRIO_T2 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/mouth_ROI_T2_56_6090_40_40pixels/'
        + 'mouth_ROI_T2_56_6090_40_40pixels.npy') / 255
    data_set_mouthRIO_T1 = img_dowm_sample(data_set_mouthRIO_T1)
    data_set_mouthRIO_T2 = img_dowm_sample(data_set_mouthRIO_T2)
    data_set_mouthRIO = np.concatenate((data_set_mouthRIO_T1, data_set_mouthRIO_T2), axis=0)  # (56, 6090, 40, 40, 3)
    data_set_mouthRIO = data_set_mouthRIO.transpose((0, 1, 4, 2, 3))   #  (56, 2030, 3, 40, 40)
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
    data_set_eyeLeftRIO_T1 = img_dowm_sample(data_set_eyeLeftRIO_T1)
    data_set_eyeLeftRIO_T2 = img_dowm_sample(data_set_eyeLeftRIO_T2)
    data_set_eyeLeftRIO = np.concatenate((data_set_eyeLeftRIO_T1, data_set_eyeLeftRIO_T2), axis=0)
    data_set_eyeLeftRIO = data_set_eyeLeftRIO.transpose((0, 1, 4, 2, 3))
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
    data_set_eyeRightRIO_T1 = img_dowm_sample(data_set_eyeRightRIO_T1)
    data_set_eyeRightRIO_T2 = img_dowm_sample(data_set_eyeRightRIO_T2)
    data_set_eyeRightRIO = np.concatenate((data_set_eyeRightRIO_T1, data_set_eyeRightRIO_T2), axis=0)
    data_set_eyeRightRIO = data_set_eyeRightRIO.transpose((0, 1, 4, 2, 3))
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
    ippg_forehead_T1 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/data_set_ippg/'
        + 'ippg_forehead_T1.npy')  # (56, 9, 6090)
    ippg_forehead_T2 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/data_set_ippg/'
        + 'ippg_forehead_T2.npy')  # (56, 9, 6090)
    data_set_ippg_forehead = np.concatenate((ippg_forehead_T1, ippg_forehead_T2), axis=0)
    print('data_set_ippg_forehead.shape = ', data_set_ippg_forehead.shape)  # (112, 9, 2030)

    del ippg_forehead_T1, ippg_forehead_T2

    # # 6. nose iPPG
    ippg_nose_T1 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/data_set_ippg/'
        + 'ippg_nose_T1.npy')  # (56, 9, 2030)
    ippg_nose_T2 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/data_set_ippg/'
        + 'ippg_nose_T2.npy')  # (56, 9, 2030)
    data_set_ippg_nose = np.concatenate((ippg_nose_T1, ippg_nose_T2), axis=0)
    print('data_set_ippg_nose.shape = ', data_set_ippg_nose.shape)  # (112, 9, 2030)
    del ippg_nose_T1, ippg_nose_T2

    # ippg
    iPPG = np.concatenate((data_set_ippg_forehead, data_set_ippg_nose), axis=1)  # (112, 18, 2030)
    iPPG = iPPG.transpose((0, 2, 1))  # (112, 6090, 18)
    print('iPPG.shape = ', iPPG.shape)

    # 划分数据集
    # 标签
    label, label_index_anxiety, label_index_anxiety_free = onehot(label)  # 转为onehot编码
    y_labels_train, y_labels_val = train_test_split(label, test_size=test_size,
                                                    random_state=random_state)
    y_labels_train = np.array(y_labels_train)
    y_labels_val = np.array(y_labels_val)
    # y_labels_train = binary_onehot(y_labels_train)
    # y_labels_val = binary_onehot(y_labels_val)
    label_train_onehot_free = y_labels_train
    label_test_onehot_free = y_labels_val
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
    #                                                            random_input_size_fc1state=random_state)
    #
    # # nose ippg
    # data_set_nose_iPPG_train, data_set_nose_iPPG_test = train_test_split(data_set_ippg_nose, test_size=test_size,
    #                                                            random_state=random_state)


    # initiates model and loss
    model = SKNet50().to(device)
    criterion = nn.BCELoss()  # https://blog.csdn.net/weixin_40522801/article/details/106616564
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ###
    total_step = len(data_set_mouthRIO_train) // batch_size
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
            # print('i = ', i)
            if i < total_step - 1:
                # print('#################')
                # print('y_labels_train.shape = ',y_labels_train.shape)
                labels = torch.from_numpy(y_labels_train[i * batch_size:(i + 1) * batch_size]).to(device)
                label_train_NO_onehot = torch.from_numpy(
                    label_train_onehot_free[i * batch_size:(i + 1) * batch_size]).to(device)
                input_shape_mouth = torch.from_numpy(
                    data_set_mouthRIO_train[i * batch_size:(i + 1) * batch_size, :]).to(device).float()
                input_shape_eyeLeft = torch.from_numpy(
                    data_set_eyeLeftRIO_train[i * batch_size:(i + 1) * batch_size]).to(device).float()
                input_shape_eyeRight = torch.from_numpy(
                    data_set_eyeRightRIO_train[i * batch_size:(i + 1) * batch_size]).to(device).float()
                input_shape_location = torch.from_numpy(
                    data_set_location_train[i * batch_size:(i + 1) * batch_size]).to(device).float()
                input_shape_iPPG = torch.from_numpy(data_set_iPPG_train[i * batch_size:(i + 1) * batch_size]).to(
                    device).float()

                # input_shape_forehead = torch.from_numpy(data_set_forehead_train[i * batch_size:(i + 1) * batch_size, :]).to(device).float()
            else:
                labels = torch.from_numpy(y_labels_train[i * batch_size:-1]).to(device)
                label_train_NO_onehot = torch.from_numpy(label_train_onehot_free[i * batch_size:-1]).to(device)
                input_shape_mouth = torch.from_numpy(data_set_mouthRIO_train[i * batch_size:-1]).to(device).float()
                input_shape_eyeLeft = torch.from_numpy(data_set_eyeLeftRIO_train[i * batch_size:-1]).to(device).float()
                input_shape_eyeRight = torch.from_numpy(data_set_eyeRightRIO_train[i * batch_size:-1]).to(
                    device).float()
                input_shape_location = torch.from_numpy(data_set_location_train[i * batch_size:-1]).to(device).float()
                input_shape_iPPG = torch.from_numpy(data_set_iPPG_train[i * batch_size:-1]).to(device).float()
                # input_shape_forehead = torch.from_numpy(data_set_forehead_train[i * batch_size:-1]).to(device).float()
            # Forward pass
            optimizer.zero_grad()
            model.train()
            with torch.no_grad():
                outputs = model(
                    input_shape_mouth,
                    input_shape_eyeLeft, input_shape_eyeRight,
                    input_shape_location,
                    input_shape_iPPG,
                )
                loss = F.cross_entropy(outputs, labels)
                loss = loss.requires_grad_()
                loss.backward()
                optimizer.step()

                if (i + 1) % 32 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}'
                          .format(epoch + 1, no_epochs, i + 1, total_step, loss.item()))
                    current_losses.append(loss.item())  # appends the current value of the loss into a list

                    y_pred, y_true, y_pred_probability, acc = validate(device, batch_size, model,
                                                                       y_labels_val,
                                                                       data_set_mouthRIO_test,
                                                                       data_set_eyeLeftRIO_test,
                                                                       data_set_eyeRightRIO_test,
                                                                       data_set_location_test,
                                                                       data_set_iPPG_test,
                                                                       training_proc_avg, test_proc_avg, last=False)

                    if acc > temp_acc:
                        temp_acc = acc
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
                        save_name = '_' + 'epochs' + str(
                            no_epochs) + '_' + 'mouth' + '_' + 'eyeLeft' + '_' + 'eyeRight' + '_' + 'location' + '_' + 'iPPG'
                        # + '_' + 'mouth' \
                        # + '_' + 'eyeLeft' + '_' + 'eyeRight' \
                        # + '_' + 'location'\
                        # + '_' + 'forehead'\
                        # + '_' + 'iPPG' # 额头提取iPPG
                        data_result = []
                        data_result.append([acc, roc_auc, precision, recall1, sensitivity, fscore1, specificity])

                        np.savetxt('./sknet_data_result_40_40pixel/sknet_data_result' + save_name + '.txt',
                                   np.array(data_result))
                        np.savetxt('./sknet_data_result_40_40pixel/sknet_y_test' + save_name + '.txt', y_true)
                        np.savetxt('./sknet_data_result_40_40pixel/sknet_y_pred_probability' + save_name + '.txt',
                                   y_pred_probability)
                        np.savetxt('./sknet_data_result_40_40pixel/sknet_y_pred' + save_name + '.txt', y_pred)
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





if __name__=='__main__':
    train()
    # x = torch.rand(8, 3, 224, 224)
    # x = torch.rand(8, 3, 40, 40)
    # model = SKNet50()
    # out = model(x)
    # print('out.shape = ', out.shape)
    #flops, params = profile(model, (x, ))
    #flops, params = clever_format([flops, params], "%.5f")
    
    #print(flops, params)
    #print('out shape : {}'.format(out.shape))

