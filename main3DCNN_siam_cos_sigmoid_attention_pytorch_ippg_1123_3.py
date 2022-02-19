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
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
print('torch.cuda.device_count()=', torch.cuda.device_count())
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

def validate(device, batch_size, model,sia_network,criterion,
                             y_labels_val,
                             data_set_mouthRIO_test,
                             data_set_eyeLeftRIO_test, data_set_eyeRightRIO_test,
                              data_set_location_test,
                             data_set_iPPG_test
                              ):
    # Test the model (validation set)
    model.eval()
    sia_network.eval()
    gen_signals_val = data_set_location_test
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
                input_shape_loc = torch.from_numpy(data_set_location_test[i * batch_size:(i + 1) * batch_size]).to(device).float()
                input_shape_iPPG = torch.from_numpy(data_set_iPPG_test[i * batch_size:(i + 1) * batch_size]).to(device).float()

                # input_shape_eTou = torch.from_numpy(data_set_eTouRIO_test[i * batch_size:(i + 1) * batch_size]).to(device).float()
            else:
                target = torch.from_numpy(y_labels_train[i * batch_size:-1]).to(device)
                input_shape_mouth = torch.from_numpy(data_set_mouthRIO_test[i * batch_size:-1]).to(device).float()
                input_shape_eyeLeft = torch.from_numpy(data_set_eyeLeftRIO_test[i * batch_size:-1]).to(device).float()
                input_shape_eyeRight = torch.from_numpy(data_set_eyeRightRIO_test[i * batch_size:-1]).to(device).float()
                input_shape_loc = torch.from_numpy(data_set_location_test[i * batch_size:-1]).to(device).float()
                input_shape_iPPG = torch.from_numpy(data_set_iPPG_test[i * batch_size:-1]).to(device).float()
                # input_shape_eTou = torch.from_numpy(data_set_eTouRIO_test[i * batch_size:-1]).to(device).float()
            # Forward pass

            out, out_fetures_fc_mouth_eyes,  out_loc, out_iPPG = model(
                input_shape_mouth,
                input_shape_eyeLeft, input_shape_eyeRight,
                input_shape_loc,
                input_shape_iPPG,
                # input_shape_eTou,
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
            # print('out = ', out)
            # print('target = ', target)
            # loss = criterion(out, target)
            # Loss = loss+ out_sia_network
            loss = F.cross_entropy(out, target)
            total_loss += loss.item()
            # total_loss += Loss.item()
            # accuracy += (torch.argmax(out, dim=1) == target).sum().item()
            # y_true.extend(torch.argmax(target, dim=1).cpu().numpy())
            # y_pred.extend(torch.argmax(out, dim=1).cpu().numpy())
            # # y_pred_probability_arr.append(out.cpu().numpy())
            # y_pred_probability.extend(
            #     torch.nn.functional.softmax(out, dim=1).cpu().detach().numpy())  # 计算softmax，即该文本属于各类的概率
            accuracy += (torch.argmax(out, dim=1) == target).sum().item()
            y_true.extend(target.cpu().numpy())
            y_pred.extend(torch.argmax(out, dim=1).cpu().numpy())
            y_pred_probability.extend(F.sigmoid(out).cpu().detach().numpy())  # 计算softmax，即属于各类的概率

            # print('out = ', out)
            # print('target = ', target)
            # print('torch.argmax(out, dim=1)  =', torch.argmax(out, dim=1))

            # y_true.extend(target.cpu().numpy())
            # y_pred_probability.extend(out.cpu().numpy())
            # print('out.shape = ', out.shape)
            # print('target.shape = ',target.shape)

    # y_pred_probability_arr = np.array(y_pred_probability_arr)
    # print('y_pred_probability_arr.shape = ', y_pred_probability_arr.shape)
    # for i in range(y_pred_probability_arr.shape[0]):
    #     # print('y_pred_probability_arr[i].shape = ', y_pred_probability_arr[i].shape)
    #     # print('y_pred_probability_arr[i] = ', y_pred_probability_arr[i])
    #     for j in range(len(y_pred_probability_arr[i])):
    #         max_pred_probability = np.max(y_pred_probability_arr[i][j])
    #         y_pred_probability.append(max_pred_probability)
    #         print('y_pred_probability_arr[i][j] = ', y_pred_probability_arr[i][j])
    #         print('max_pred_probability = ', max_pred_probability)

    y_pred_probability = np.array(y_pred_probability)
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
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn_Siamese = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=2),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=2),
            nn.MaxPool2d(2, stride=2))
        ##### input_size_fc_Siamese需要重新设置
        input_size_fc_Siamese =  3100
        output_size_fc_Siamese = 25
        self.fc_Siamese = nn.Sequential(
            nn.Linear(input_size_fc_Siamese, output_size_fc_Siamese),
            nn.ReLU(inplace=True),
            nn.Linear(output_size_fc_Siamese, 10),
            nn.Linear(10, 1))

    def forward_once(self, x):
        output = self.cnn_Siamese(x)
        output = output.view(output.size()[0], -1)
        # print('output.shape = ', output.shape)
        output = self.fc_Siamese(output)
        return output

    def similarity_calculate(self, output1, output2):
        # print('output1 = ', output1)
        # print('output2 = ', output2)
        # 计算相似度:欧式距离 https://cloud.tencent.com/developer/article/1668762
        sum_dis = 0.0
        # 欧式距离
        # for i in range(len(output1)):
        #     dis = output1[i]-output2[i]
        #     sum_dis = sum_dis + dis*dis
        # # print('sum_dis = ', sum_dis)
        # similarity = sum_dis** 0.5

        # 曼哈顿距离  abs()
        # for i in range(len(output1)):
        #     dis = abs(output1[i]-output2[i])
        #     sum_dis = sum_dis + dis * dis
        # similarity = sum_dis

        # 求余弦相似度
        dis_output1 = 0.0
        dis_output2 = 0.0
        for i in range(len(output1)):
            # 求分子
            dis = output1[i] * output2[i]
            sum_dis = sum_dis + dis
            # 求分母
            dis_output1 = dis_output1 + output1[i] * output1[i]
            dis_output2 = dis_output2 + output2[i] * output2[i]

        sim = sum_dis / ((dis_output1 ** 0.5) * (dis_output2 ** 0.5))  # 范围在[-1,1]
        # print('sim = ', sim)
        # 转换至0-1区间
        similarity = (sim - (-1)) / (1 - (-1))
        # print('similarity = ', similarity)
        return  similarity

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        # print('len(output1)=',len(output1))
        # print('output1.shape = ', output1.shape)
        # print('output2.shape = ', output2.shape)
        similarity = self.similarity_calculate(output1, output2)
        return similarity

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
        features_num = 64
        self.ca = ChannelAttention(features_num)
        self.sa = SpatialAttention()
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
        self.n_units = 64
        # fc_input = 32 * 40 * 1 * 95
        fc_input =  32*40*1*63

        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 4, 2), ceil_mode=False)
        self.fc = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(6400, 1024),
            # nn.ReLU(True),
            # nn.Sigmoid(),
            # nn.Linear(1024, 512),
            # nn.Linear(512, 256),
            # nn.Linear(256, 128),
            # nn.Linear(128, num_classes),

            # nn.Linear(32*40*1*93,  self.n_units),# 3000张
            # nn.Linear(32 * 40 * 1 * 190, self.n_units), #6090张
            nn.Linear(fc_input, self.n_units),
        )

        ## LSTM
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
        self.out = nn.Linear(self.n_units, self.num_classes)

        # self.fc1 = nn.Sequential(nn.Linear(2*2, self.num_classes))
        num_features = 5 # 根据输入特征的个数进行修改
        self.fc_sum = nn.Sequential(
                                    nn.Linear(self.n_units * num_features, self.num_classes),
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
        out = self.ca(out) * out
        out = self.sa(out) * out
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
        out_pool3 = self.pool3(out)
        # print('out.shape = ', out.shape)
        # out_pool3 = out.transpose((0, 1, 3, 2, 4))  # 数据维度变换
        # print('out_pool3.shape = ', out_pool3.shape)
        out = self.fc(out_pool3)
        return out, out_pool3

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
                input_shape_loc,
                input_shape_iPPG,
                # input_shape_eTou,
                ):


        features_mouth, features_mouth_out_pool3 = self.feature_extract_3DCNN(input_shape_mouth)
        features_eyeLeft, features_eyeLeft_out_pool3 = self.feature_extract_3DCNN(input_shape_eyeLeft)
        features_eyeRight, features_eyeRight_out_pool3 = self.feature_extract_3DCNN(input_shape_eyeRight)
        features_loc = self.feature_extract_LSTM(input_shape_loc)
        features_iPPG = self.feature_extract_iPPG(input_shape_iPPG)
        # features_eTou, features_eTou_out_pool3 = self.feature_extract_3DCNN(input_shape_eTou)
        # out = features_imgface
        out_fetures = torch.cat((
            features_mouth,
            features_eyeLeft, features_eyeRight,
            features_loc,
            features_iPPG,
            ), 1)  # 在 1 维(横向)进行拼接

        # 嘴巴和眼睛串联
        out_fetures_fc_mouth_eyes = torch.cat((
            features_mouth_out_pool3,
            features_eyeLeft_out_pool3, features_eyeRight_out_pool3,
            # features_eTou_out_pool3,
            ), 1)  # 在 1 维(横向)进行拼接

        # print('features_eTou_out_pool3.shape = ', features_eTou_out_pool3.shape)
        # print('out_fetures_fc.shape = ', out_fetures_fc.shape)
        # print('out.shape = ', out.shape)
        out = self.fc_sum(out_fetures)
        return out, out_fetures_fc_mouth_eyes, features_loc,features_iPPG


# https://zhuanlan.zhihu.com/p/99261200
# 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        #
        avg_out = avg_out = self.avg_pool(x)
        avg_out = self.fc1(avg_out)
        avg_out = self.relu1(avg_out)
        avg_out = self.fc2(avg_out)
        ##
        max_out = self.max_pool(x)
        max_out = self.fc1(max_out)
        max_out = self.relu1(max_out)
        max_out = self.fc2(max_out)
        out = avg_out + max_out
        return self.sigmoid(out)


# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

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
            if j%img_num_per_second==0:
                data_arr.append(data[i,j])
    data_arr = np.reshape(data_arr,(data.shape[0], int(data.shape[1]/img_num_per_second),data.shape[2],data.shape[3],data.shape[4]))
    del data
    return data_arr


# 计算相似度
def similar_cal(output1, output2):
    # print('output1.shape = ', output1.shape)
    # print('output2.shape = ', output2.shape)
    # print('len(output1) = ', len(output1))
    # 计算相似度:欧式距离 https://cloud.tencent.com/developer/article/1668762
    sum_dis = 0.0
    # 欧式距离
    # for i in range(len(output1)):
    #     dis = output1[i]-output2[i]
    #     sum_dis = sum_dis + dis*dis
    # # print('sum_dis = ', sum_dis)
    # similarity = sum_dis** 0.5

    # 曼哈顿距离  abs()
    # for i in range(len(output1)):
    #     dis = abs(output1[i]-output2[i])
    #     sum_dis = sum_dis + dis * dis
    # similarity = sum_dis

    # 求余弦相似度
    dis_output1 = 0.0
    dis_output2 = 0.0
    for i in range(output2.shape[0]):
        # 求分子
        dis = output1[i] * output2[i]
        sum_dis = sum_dis + dis
        # 求分母
        dis_output1 = dis_output1 + output1[i] * output1[i]
        dis_output2 = dis_output2 + output2[i] * output2[i]

    sim = sum_dis / ((dis_output1 ** 0.5) * (dis_output2 ** 0.5))  # 范围在[-1,1]
    # print('sim = ', sim)
    # 转换至0-1区间
    similarity = (sim - (-1)) / (1 - (-1))
    # print('similarity = ', similarity)
    return similarity

# 特征转换成需要的输入
def features_transmit_mouth_eyes(data):
    data = np.reshape(data,(1,
                            data.shape[0],
                            data.shape[1],
                            data.shape[2],
                            data.shape[3]))
    return data

def features_transmit_ippg_loc(data):
    data = np.reshape(data, (1,
                             data.shape[0],
                             data.shape[1]))
    return data

def siamese_input_reshape(data): # input size [96, 1, 252, 10]
    # print('****data.shape=', data.shape) # [96, 40, 1, 63]  注意out1和out2的维度是不一样的：out1是5维，out2是4维
    data = torch.reshape(data,(96,  # 96
                               1,  # 1
                               int(63*40/10), #
                               10))
    return data

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

    del  ippg_forehead_T1, ippg_forehead_T2

    # # 6. nose iPPG
    ippg_nose_T1 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/data_set_ippg/'
        + 'ippg_nose_T1.npy')  # (56, 9, 2030)
    ippg_nose_T2 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/data_set_ippg/'
        + 'ippg_nose_T2.npy')  # (56, 9, 2030)
    data_set_ippg_nose = np.concatenate((ippg_nose_T1, ippg_nose_T2), axis=0)
    print('data_set_ippg_nose.shape = ', data_set_ippg_nose.shape)  # (112, 9, 2030)
    del  ippg_nose_T1, ippg_nose_T2

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

    # del data_set_mouthRIO,

    # initiates model and loss
    model = ConvNet3D_one(no_classes).to(device)
    # model_ConvNet3D_two = ConvNet3D_two(no_classes).to(device)
    sia_network = SiameseNetwork().to(device)
    # criterion = nn.CrossEntropyLoss()  # alternatively MSE if regression or PSNR/PSD or pearson correlation
    criterion = nn.BCELoss()  # https://blog.csdn.net/weixin_40522801/article/details/106616564
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer_sia_network = torch.optim.Adam(sia_network.parameters(), lr=learning_rate)
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
                label_train_NO_onehot = torch.from_numpy(label_train_onehot_free[i * batch_size:(i + 1) * batch_size]).to(device)
                input_shape_mouth = torch.from_numpy(data_set_mouthRIO_train[i * batch_size:(i + 1) * batch_size, :]).to(device).float()
                input_shape_eyeLeft = torch.from_numpy(data_set_eyeLeftRIO_train[i * batch_size:(i + 1) * batch_size]).to(device).float()
                input_shape_eyeRight = torch.from_numpy(data_set_eyeRightRIO_train[i * batch_size:(i + 1) * batch_size]).to(device).float()
                input_shape_location = torch.from_numpy(data_set_location_train[i * batch_size:(i + 1) * batch_size]).to(device).float()
                input_shape_iPPG = torch.from_numpy(data_set_iPPG_train[i * batch_size:(i + 1) * batch_size]).to(device).float()

                # input_shape_forehead = torch.from_numpy(data_set_forehead_train[i * batch_size:(i + 1) * batch_size, :]).to(device).float()
            else:
                labels = torch.from_numpy(y_labels_train[i * batch_size:-1]).to(device)
                label_train_NO_onehot = torch.from_numpy(label_train_onehot_free[i * batch_size:-1]).to(device)
                input_shape_mouth = torch.from_numpy(data_set_mouthRIO_train[i * batch_size:-1]).to(device).float()
                input_shape_eyeLeft = torch.from_numpy(data_set_eyeLeftRIO_train[i * batch_size:-1]).to(device).float()
                input_shape_eyeRight = torch.from_numpy(data_set_eyeRightRIO_train[i * batch_size:-1]).to(device).float()
                input_shape_location = torch.from_numpy(data_set_location_train[i * batch_size:-1]).to(device).float()
                input_shape_iPPG = torch.from_numpy(data_set_iPPG_train[i * batch_size:-1]).to(device).float()
                # input_shape_forehead = torch.from_numpy(data_set_forehead_train[i * batch_size:-1]).to(device).float()
            # Forward pass
            optimizer.zero_grad()
            optimizer_sia_network.zero_grad()
            model.train()
            sia_network.train()
            with torch.no_grad():
                out1,out1_fetures_fc_mouth_eyes, out1_loc, out1_iPPG = model(
                              input_shape_mouth,
                              input_shape_eyeLeft, input_shape_eyeRight,
                              input_shape_location,
                              input_shape_iPPG,
                              )


                loss_eye_mouth_arr = []
                loss_mouth_arr =[]
                loss_eyeLeft_arr  =[]
                loss_eyeRight_arr = []
                loss_loc_arr = []
                loss_ippg_arr = []

                for m in range(out1.shape[0]):
                    # print('m =', m)
                    # 从焦虑人群中随机抽取anxiety_num个人，计算相似度
                    anxiety_num = 5
                    for n in range(anxiety_num):
                        # 随机取索引
                        rand = int(random.random()*len(label_index_anxiety))
                        rand_index = label_index_anxiety[rand]
                        # print('rand_index=  ',rand_index)
                        # 根据随机抽取的索引，抽取对应的原始特征
                        input2_mouth_anxiety =  features_transmit_mouth_eyes(data_set_mouthRIO[rand_index])
                        input2_eyeLeft_anxiety = features_transmit_mouth_eyes(data_set_eyeLeftRIO[rand_index])
                        input2_eyeRight_anxiety = features_transmit_mouth_eyes(data_set_eyeRightRIO[rand_index])
                        input2_location_anxiety = features_transmit_ippg_loc(location[rand_index])
                        input2_iPPG_anxiety =  features_transmit_ippg_loc(iPPG[rand_index])

                        # 转换成tensor数据类型
                        input2_mouth_anxiety = torch.from_numpy(input2_mouth_anxiety).to(device).float()
                        input2_eyeLeft_anxiety = torch.from_numpy(input2_eyeLeft_anxiety).to(device).float()
                        input2_eyeRight_anxiety = torch.from_numpy(input2_eyeRight_anxiety).to(device).float()
                        input2_location_anxiety = torch.from_numpy(input2_location_anxiety).to(device).float()
                        input2_iPPG_anxiety = torch.from_numpy(input2_iPPG_anxiety).to(device).float()

                        # 经过特征提取网络进行特征提取
                        # out, out_fetures_fc_mouth_eyes, features_loc, features_iPPG
                        out2, out2_fetures_fc_mouth_eyes, out2_loc, out2_iPPG= model(
                            input2_mouth_anxiety,
                            input2_eyeLeft_anxiety, input2_eyeRight_anxiety,
                            input2_location_anxiety,
                            input2_iPPG_anxiety,
                        )
                        # 计算相似度
                        # print('out1_mouth[m].shape = ', out1_mouth[m].shape)
                        # print('out2_mouth.shape = ', out2_mouth.shape)
                        # print('out1_eyeLeft[m].shape = ', out1_eyeLeft[m].shape)
                        # print('out2_eyeLeft.shape = ', out2_eyeLeft.shape)
                        # print('out1_eyeRight[m].shape=', out1_eyeRight[m].shape)
                        # print('out2_eyeRight.shape = ', out2_eyeRight.shape)
                        # print('out1_loc[m].shape = ', out1_loc[m].shape)
                        # print('out2_loc.shape =', out2_loc.shape)
                        # print('out1_iPPG[m]=',out1_iPPG[m].shape)
                        # print('out2_iPPG.shape = ', out2_iPPG.shape)
                        # a = siamese_input_reshape(out1_mouth[m])
                        # b = siamese_input_reshape(out2_mouth)
                        # sim_mouth = sia_network(siamese_input_reshape(out1_mouth[m]), siamese_input_reshape(out2_mouth))
                        # sim_eyeLeft = sia_network(siamese_input_reshape(out1_eyeLeft[m]), siamese_input_reshape(out2_eyeLeft))
                        # sim_eyeRight = sia_network(siamese_input_reshape(out1_eyeRight[m]), siamese_input_reshape(out2_eyeRight))
                        # sim_location = sia_network(siamese_input_reshape(out1_loc[m]), siamese_input_reshape(out2_loc))
                        # sim_iPPG = sia_network(siamese_input_reshape(out1_iPPG[m]), siamese_input_reshape(out2_iPPG))
                        input1_siamese = siamese_input_reshape(out1_fetures_fc_mouth_eyes[m])
                        input2_siamese = siamese_input_reshape(out2_fetures_fc_mouth_eyes)
                        sim_eye_mouth= sia_network(input1_siamese, input2_siamese)
                        sim_loc = similar_cal(out1_loc[m], out2_loc)
                        sim_ippg = similar_cal(out1_iPPG[m], out2_iPPG)

                        # loss_sim
                        # loss_mouth = abs(sim_mouth-labels[m])
                        # loss_eyeLeft = abs(sim_eyeLeft - labels[m])
                        # loss_eyeRight = abs(sim_eyeRight - labels[m])
                        # loss_loc = abs(sim_location - labels[m])
                        # loss_ippg = abs(sim_iPPG - labels[m])
                        loss_eye_mouth =  abs(sim_eye_mouth - labels[m])
                        loss_loc = abs(sim_loc - labels[m])
                        loss_ippg = abs(sim_ippg - labels[m])
                        #
                        # loss_mouth_arr.append(loss_mouth.data.cpu().numpy())
                        # loss_eyeLeft_arr.append(loss_eyeLeft.data.cpu().numpy())
                        # loss_eyeRight_arr.append(loss_eyeRight.data.cpu().numpy())
                        # loss_loc_arr.append(loss_loc.data.cpu().numpy())
                        # loss_ippg_arr.append(loss_ippg.data.cpu().numpy())
                        loss_eye_mouth_arr.append(loss_eye_mouth.data.cpu().numpy())
                        loss_loc_arr.append(loss_loc.data.cpu().numpy())
                        loss_ippg_arr.append(loss_ippg.data.cpu().numpy())

                # 转换成数组
                # loss_mouth_arr = np.array(loss_mouth_arr)
                # loss_eyeLeft_arr = np.array(loss_eyeLeft_arr)
                # loss_eyeRight_arr = np.array(loss_eyeRight_arr)
                # loss_loc_arr = np.array(loss_loc_arr)
                # loss_ippg_arr = np.array(loss_ippg_arr)
                # 计算Loss
                loss = F.cross_entropy(out1, labels)
                # Loss = 0.5*(np.mean(loss_mouth_arr) + np.mean(loss_eyeLeft_arr)
                #              + np.mean(loss_eyeRight_arr)
                #              + np.mean(loss_loc_arr) + np.mean(loss_ippg_arr)) + 0.5*loss

                Loss = 0.1*(np.mean(loss_eye_mouth_arr) + np.mean(loss_loc_arr)
                             + np.mean(loss_ippg_arr)) + 0.9*loss
                Loss = Loss.requires_grad_()
                Loss.backward()
                optimizer.step()
                print('Loss =', Loss)
                if (i + 1) % 32 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}'
                          .format(epoch + 1, no_epochs, i + 1, total_step, loss.item()))
                    current_losses.append(loss.item())  # appends the current value of the loss into a list

                    y_pred, y_true, y_pred_probability,acc = validate(device, batch_size, model,sia_network,criterion,
                             y_labels_val,
                             data_set_mouthRIO_test,
                             data_set_eyeLeftRIO_test, data_set_eyeRightRIO_test,
                              data_set_location_test,
                             data_set_iPPG_test)
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
                            no_epochs) + '_' + 'mouth' + '_' + 'eyeLeft' + '_' + 'eyeRight' + '_' + 'location'+ '_' + 'iPPG'
                        # + '_' + 'mouth' \
                        # + '_' + 'eyeLeft' + '_' + 'eyeRight' \
                        # + '_' + 'location'\
                        # + '_' + 'forehead'\
                        # + '_' + 'iPPG' # 额头提取iPPG
                        data_result = []
                        data_result.append([acc, roc_auc, precision, recall1, sensitivity, fscore1, specificity])
                        # np.savetxt('resnet_3D_fpr' + save_name + '.txt', fpr)
                        # np.savetxt('resnet_3D_tpr' + save_name + '.txt', tpr)
                        np.savetxt(
                            './data_result_ours_attention_40_40pixel_1123/3DCNN_sigmoid_attention_data_result_3' + save_name + '.txt',
                            data_result)
                        np.savetxt(
                            './data_result_ours_attention_40_40pixel_1123/3DCNN_sigmoid_attention_y_test_3' + save_name + '.txt',
                            y_true)
                        np.savetxt(
                            './data_result_ours_attention_40_40pixel_1123/3DCNN_sigmoid_attention_y_pred_probability_3' + save_name + '.txt',
                            y_pred_probability)
                        np.savetxt(
                            './data_result_ours_attention_40_40pixel_1123/3DCNN_sigmoid_attention_sigmoid_y_pred_3' + save_name + '.txt',
                            y_pred)

if __name__ == '__main__':
    train()