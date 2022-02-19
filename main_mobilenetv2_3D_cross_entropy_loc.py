'''MobilenetV2 in PyTorch.

See the paper "MobileNetV2: Inverted Residuals and Linear Bottlenecks" for more details.
'''
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
            accuracy += (torch.argmax(out, dim=1) == target).sum().item()
            # y_true.extend(torch.argmax(target, dim=1).cpu().numpy())
            # y_pred.extend(torch.argmax(out, dim=1).cpu().numpy())
            # y_pred_probability_arr.append(out.cpu().numpy())
            # print('out = ', out)
            # print('target = ', target)
            # print('torch.argmax(out, dim=1)  =', torch.argmax(out, dim=1))


            y_true.extend(target.cpu().numpy())
            y_pred.extend(torch.argmax(out, dim=1).cpu().numpy())
            # y_pred_probability_arr.append(out.cpu().numpy())
            y_pred_probability.extend(F.sigmoid(out).cpu().detach().numpy())  # 计算softmax，即属于各类的概率


            # y_true.extend(target.cpu().numpy())
            # y_pred_probability.extend(out.cpu().numpy())
            # print('out.shape = ', out.shape)
            # print('target.shape = ',target.shape)

    # y_pred_probability_arr = np.array(y_pred_probability_arr)
    # # print('y_pred_probability_arr.shape = ', y_pred_probability_arr.shape)
    # for i in range(y_pred_probability_arr.shape[0]):
    #     # print('y_pred_probability_arr[i].shape = ', y_pred_probability_arr[i].shape)
    #     # print('y_pred_probability_arr[i] = ', y_pred_probability_arr[i])
    #     for j in range(len(y_pred_probability_arr[i])):
    #         max_pred_probability = np.max(y_pred_probability_arr[i][j])
    #         y_pred_probability.append(max_pred_probability)
    #         # print('y_pred_probability_arr[i][j] = ', y_pred_probability_arr[i][j])
    #         # print('max_pred_probability = ', max_pred_probability)

    y_pred_probability = np.array(y_pred_probability)
    print('y_pred_probability  = ', y_pred_probability)
    print('##############')
    y_pred_probability = y_pred_probability[:, 1]
    print('y_pred_probability  = ', y_pred_probability)
    acc = accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    print('acc = ', acc)
    # print('y_true = ', y_true)
    # print('y_pred_probability = ', y_pred_probability)
    # print('np.array(y_pred).shape = ', np.array(y_pred).shape)
    # print('np.array(y_true).shape = ', np.array(y_true).shape)
    # print('np.array(y_pred_probability).shape = ', np.array(y_pred_probability).shape)
    # print('np.array(y_labels_val).shape = ', np.array(y_labels_val).shape)

    return y_pred, y_true, y_pred_probability, acc


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=2, sample_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1,  16, 1, (1,1,1)],
            [6,  24, 2, (2,2,2)],
            [6,  32, 3, (2,2,2)],
            [6,  64, 4, (2,2,2)],
            [6,  96, 3, (1,1,1)],
            [6, 160, 3, (2,2,2)],
            [6, 320, 1, (1,1,1)],
        ]

        # building first layer
        assert sample_size % 16 == 0.
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, (1,2,2))]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else (1,1,1)
                self.features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )
        ##
        self.n_units = 128
        input_fc_picture = 1280
        # output_fc_picture = 128
        self.fc_picture = nn.Sequential(
            nn.Linear(input_fc_picture, self.n_units),
            # nn.Sigmoid(), # 和nn.BCELoss()配合使用，详见https://blog.csdn.net/weixin_40522801/article/details/106616564
        )
        ##################
        num_features = 4  # 根据输入特征的个数进行修改
        self.fc_sum = nn.Sequential(
            nn.Linear(num_features * 128, num_classes),
            # nn.Sigmoid(), # 和nn.BCELoss()配合使用，详见https://blog.csdn.net/weixin_40522801/article/details/106616564
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
        ##########################
        self._initialize_weights()
    def feature_extract_LSTM(self, x):
        # print('x.shape= ', x.shape)
        x = x.to(device).float()
        r_out, (h_n, h_c) = self.rnn(x, None)
        # out = self.out(r_out[:, -1, :])
        out = r_out[:, -1, :]
        return out

    def feature_extract(self, x):
        x = self.features(x)
        x = F.avg_pool3d(x, x.data.size()[-3:])
        x = x.view(x.size(0), -1)
        x =  self.fc_picture(x)
        return x
    def forward(self,
                input_shape_mouth,
                input_shape_eyeLeft, input_shape_eyeRight,
                input_shape_xyz,
                # input_shape_eTou,
                ):


        features_mouth = self.feature_extract(input_shape_mouth)
        features_eyeLeft = self.feature_extract(input_shape_eyeLeft)
        features_eyeRight = self.feature_extract(input_shape_eyeRight)
        features_xyz = self.feature_extract_LSTM(input_shape_xyz)
        # features_eTou = self.feature_extract(input_shape_eTou)

        ##############################################################

        # print('features_mouth.shape = ', features_mouth.shape)
        # print('features_xyz.shape = ', features_xyz.shape)
        out = torch.cat((
            features_mouth,
            features_eyeLeft, features_eyeRight,
            features_xyz,
            # features_eTou,
        ), 1)  # 在 1 维(横向)进行拼接
        # print('out.shape = ', out.shape)
        out = self.fc_sum(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1,1,1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == (1,1,1) and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


# def get_model(**kwargs):
#     """
#     Returns the model.
#     """
#     model = MobileNetV2(**kwargs)
#     return model
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


def data_downsample(index_arr, label_set,
                    data_set_mouthRIO_mod4_0,
                    data_set_eyeLeftRIO_mod4_0,
                    data_set_eyeRightRIO_mod4_0,
                    data_set_xyz,
                    data_set_eTouRIO_mod4_0):
    data_mouth_anxiety = []
    data_eyeLeft_anxiety = []
    data_eyeRightRIO_anxiety = []
    data_location_anxiety = []
    data_forehead_anxiety = []

    data_mouth_NOanxiety = []
    data_eyeLeft_NOanxiety = []
    data_eyeRightRIO_NOanxiety = []
    data_location_NOanxiety = []
    data_forehead_NOanxiety = []
    index_anxiety  =[]
    index_noanxiety =[]
    print('len(index_arr) = ', len(index_arr))
    for i in range(len(index_arr)):
        # 把焦虑人群的数据挑出来
        print('label_set[index_arr[i]]= ', label_set[index_arr[i]])
        mouth_temp = data_set_mouthRIO_mod4_0[index_arr[i]]
        eyeLeft_temp = data_set_eyeLeftRIO_mod4_0[index_arr[i]]
        eyeRight_temp = data_set_eyeRightRIO_mod4_0[index_arr[i]]
        loc_temp = data_set_xyz[index_arr[i]]
        forehead_temp = data_set_eTouRIO_mod4_0[index_arr[i]]
        print('mouth_temp.shape = ', mouth_temp.shape)
        print('eyeLeft_temp.shape = ', eyeLeft_temp.shape)
        print('eyeRight_temp.shape = ', eyeRight_temp.shape)
        print('loc_temp.shape = ', loc_temp.shape)

        if label_set[index_arr[i]]!=0:
            # 保存数据
            data_mouth_anxiety.append(mouth_temp)
            data_eyeLeft_anxiety.append(eyeLeft_temp)
            data_eyeRightRIO_anxiety.append(eyeRight_temp)
            data_location_anxiety.append(loc_temp)
            data_forehead_anxiety.append(forehead_temp)
            index_anxiety.append(label_set[index_arr[i]])

        else:
            # 保存数据
            data_mouth_NOanxiety.append(mouth_temp)
            data_eyeLeft_NOanxiety.append(eyeLeft_temp)
            data_eyeRightRIO_NOanxiety.append(eyeRight_temp)
            data_location_NOanxiety.append(loc_temp)
            data_forehead_NOanxiety.append(forehead_temp)
            index_noanxiety.append(label_set[index_arr[i]])
    data_mouth_anxiety = np.array(data_mouth_anxiety)
    data_eyeLeft_anxiety = np.array(data_eyeLeft_anxiety)
    data_eyeRightRIO_anxiety = np.array(data_eyeRightRIO_anxiety)
    data_forehead_anxiety = np.array(data_forehead_anxiety)
    data_location_anxiety = np.array(data_location_anxiety)


    data_mouth_NOanxiety = np.array(data_mouth_NOanxiety)
    data_eyeLeft_NOanxiety = np.array(data_eyeLeft_NOanxiety)
    data_eyeRightRIO_NOanxiety = np.array(data_eyeRightRIO_NOanxiety)
    data_forehead_NOanxiety = np.array(data_forehead_NOanxiety)
    data_location_NOanxiety = np.array(data_location_NOanxiety)

    print('data_mouth_anxiety.shape = ', data_mouth_anxiety.shape)
    print('data_eyeLeft_anxiety.shape = ', data_eyeLeft_anxiety.shape)
    print('data_eyeRightRIO_anxiety.shape = ', data_eyeRightRIO_anxiety.shape)
    print('data_forehead_anxiety.shape = ', data_forehead_anxiety.shape)
    print('data_forehead_anxiety.shape = ', data_forehead_anxiety.shape)
    print('data_location_anxiety.shape = ', data_location_anxiety.shape)

    print('data_mouth_NOanxiety.shape = ', data_mouth_NOanxiety.shape)
    print('data_eyeLeft_NOanxiety.shape = ', data_eyeLeft_NOanxiety.shape)
    print('data_eyeRightRIO_NOanxiety.shape = ', data_eyeRightRIO_NOanxiety.shape)
    print('data_location_NOanxiety.shape = ', data_location_NOanxiety.shape)
    print('data_forehead_NOanxiety.shape = ', data_forehead_NOanxiety.shape)
    print('data_location_NOanxiety.shape = ', data_location_NOanxiety.shape)
    num_NOanxiety = 200 - data_location_anxiety.shape[0]
    data_mouth =  np.concatenate((data_mouth_anxiety, data_mouth_NOanxiety[0:num_NOanxiety]), axis=0)
    data_eyeLeft = np.concatenate((data_eyeLeft_anxiety, data_eyeLeft_NOanxiety[0:num_NOanxiety]), axis=0)
    data_eyeRightRIO = np.concatenate((data_eyeRightRIO_anxiety, data_eyeRightRIO_NOanxiety[0:num_NOanxiety]), axis=0)
    data_location = np.concatenate((data_location_anxiety, data_location_NOanxiety[0:num_NOanxiety]), axis=0)
    data_forehead = np.concatenate((data_forehead_anxiety, data_forehead_NOanxiety[0:num_NOanxiety]), axis=0)
    label_arr = np.concatenate((index_anxiety, index_noanxiety[0:num_NOanxiety]), axis=0)


    data_mouth = data_mouth.transpose((0, 4, 1, 2, 3))
    data_eyeLeft = data_eyeLeft.transpose((0, 4, 1, 2, 3))
    data_eyeRightRIO = data_eyeRightRIO.transpose((0, 4, 1, 2, 3))
    data_forehead = data_forehead.transpose((0, 4, 1, 2, 3))
    #
    # print('data_mouth.shape = ', data_mouth.shape)
    # print('data_eyeLeft.shape = ', data_eyeLeft.shape)
    # print('data_eyeRightRIO.shape = ', data_eyeRightRIO.shape)
    # print('data_head.shape = ', data_head.shape)
    # print('data_location.shape = ', data_location.shape)
    # # 索引打乱
    # print('data_mouth.shape =', data_mouth.shape)
    # import random
    # index = []
    # for i in range(data_mouth.shape[0]):
    #     index.append(i)
    # random.shuffle(index)
    # print('index= ', index)
    # mouth = []
    # eyeLeft = []
    # eyeRightRIO = []
    # forehead = []
    # location = []
    # label = []
    # for j in range(len(index)):
    #     mouth.append(data_mouth[index[j]])
    #     eyeLeft.append(data_eyeLeft[index[j]])
    #     eyeRightRIO.append(data_eyeRightRIO[index[j]])
    #     location.append(data_location[index[j]])
    #     forehead.append(data_head[index[j]])
    #     label.append(label_arr[index[j]])
    #
    # mouth = np.array(mouth)
    # eyeLeft = np.array(eyeLeft)
    # eyeRightRIO = np.array(eyeRightRIO)
    # forehead = np.array(forehead)
    # location = np.array(location)
    # label = np.array(label)
    # mouth = mouth.transpose((0, 4, 1, 2, 3))
    # eyeLeft = eyeLeft.transpose((0, 4, 1, 2, 3))
    # eyeRightRIO = eyeRightRIO.transpose((0, 4, 1, 2, 3))
    # forehead = forehead.transpose((0, 4, 1, 2, 3))

    # return mouth, eyeLeft, eyeRightRIO, forehead, location,label
    return data_mouth, data_eyeLeft, data_eyeRightRIO, data_forehead, data_location,label_arr
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
    no_epochs = 50 # 5000 originally
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
    model = MobileNetV2(num_classes=no_classes).to(device)
    # criterion = nn.CrossEntropyLoss()  # alternatively MSE if regression or PSNR/PSD or pearson correlation
    criterion = nn.BCELoss()  # https://blog.csdn.net/weixin_40522801/article/details/106616564
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ###
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
                input_shape_mouth= torch.from_numpy(data_set_mouthRIO_train[i * batch_size:(i + 1) * batch_size,:]).to(device).float()
                input_shape_eyeLeft = torch.from_numpy(data_set_eyeLeftRIO_train[i * batch_size:(i + 1) * batch_size]).to(device).float()
                input_shape_eyeRight = torch.from_numpy(data_set_eyeRightRIO_train[i * batch_size:(i + 1) * batch_size]).to(device).float()
                input_shape_location = torch.from_numpy(data_set_location_train[i * batch_size:(i + 1) * batch_size]).to(device).float()
                # input_shape_forehead = torch.from_numpy(data_set_forehead_train[i * batch_size:(i + 1) * batch_size,:]).to(device).float()
            else:
                labels = torch.from_numpy(y_labels_train[i * batch_size:-1]).to(device)
                input_shape_mouth = torch.from_numpy(data_set_mouthRIO_train[i * batch_size:-1,:]).to(device).float()
                input_shape_eyeLeft = torch.from_numpy(data_set_eyeLeftRIO_train[i * batch_size:-1]).to(device).float()
                input_shape_eyeRight = torch.from_numpy(data_set_eyeRightRIO_train[i * batch_size:-1]).to(device).float()
                input_shape_location = torch.from_numpy(data_set_location_train[i * batch_size:-1]).to(device).float()
                # input_shape_forehead = torch.from_numpy(data_set_forehead_train[i * batch_size:-1,:]).to(device).float()
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
                # print('outputs.shape = ', outputs.shape)
                # print('labels.shape = ', labels.shape)
                # print('outputs = ',outputs)
                # loss = criterion(outputs, labels)
                loss = F.cross_entropy(outputs, labels)
                loss = loss.requires_grad_()
                # print('loss = ',loss)
                loss.backward()
                optimizer.step()
                if (i + 1) % 34 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}'
                          .format(epoch + 1, no_epochs, i + 1, total_step, loss.item()))
                    current_losses.append(loss.item())  # appends the current value of the loss into a list

                    y_pred, y_true, y_pred_probability,acc = validate(device, batch_size, model,criterion,
                             y_labels_val,
                             data_set_mouthRIO_test,
                             data_set_eyeLeftRIO_test, data_set_eyeRightRIO_test,
                              data_set_location_test,
                              # data_set_forehead_test,
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
    save_name = '_' + 'epochs' + str(no_epochs)+ '_' + 'mouth'+ '_' + 'eyeLeft' + '_' + 'eyeRight' + '_' + 'location'
    # + '_' + 'mouth' \
    # + '_' + 'eyeLeft' + '_' + 'eyeRight' \
    # + '_' + 'location'\
    # + '_' + 'forehead'\
    # + '_' + 'iPPG' # 额头提取iPPG
    data_result = []
    data_result.append([acc, roc_auc, precision, recall1, sensitivity, fscore1, specificity])
    # np.savetxt('resnet_3D_fpr' + save_name + '.txt', fpr)
    # np.savetxt('resnet_3D_tpr' + save_name + '.txt', tpr)
    np.savetxt('./data_result_mobilenetv2_3D_40_40pixel/mobilenetv2_sigmoid_data_result' + save_name + '.txt', data_result)
    np.savetxt('./data_result_mobilenetv2_3D_40_40pixel/mobilenetv2_sigmoid_y_test' + save_name + '.txt', y_true)
    np.savetxt('./data_result_mobilenetv2_3D_40_40pixel/mobilenetv2_sigmoid_y_pred_probability' + save_name + '.txt', y_pred_probability)
    np.savetxt('./data_result_mobilenetv2_3D_40_40pixel/mobilenetv2_sigmoid_y_pred' + save_name + '.txt', y_pred)

    # np.savetxt('mobilenetv2_sigmoid_NO_augmentation_data_result' + save_name + '.txt', data_result)
    # np.savetxt('mobilenetv2_sigmoid_NO_augmentation_y_test' + save_name + '.txt', y_true)
    # np.savetxt('mobilenetv2_sigmoid_NO_augmentation_y_pred_probability' + save_name + '.txt', y_pred_probability)
    # np.savetxt('mobilenetv2_sigmoid_NO_augmentation_y_pred' + save_name + '.txt', y_pred)

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
    # torch.save(model, './data_result_mobilenetv2_3D_40_40pixel/mobilenetv2_3D_mod4_0_1_anxiety_screeen' + save_name + '.pkl')


if __name__ == "__main__":
    # model = get_model(num_classes=600, sample_size=112, width_mult=1.)
    # model = model.cuda()
    # model = nn.DataParallel(model, device_ids=None)
    # print(model)
    #
    #
    # input_var = Variable(torch.randn(8, 3, 16, 112, 112))
    # output = model(input_var)
    # print(output.shape)

    train()


