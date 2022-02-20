# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
#  新服务器下的代码，mhmpy37环境
# code path: /home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen/2021-10-21-TSM/my_TSM/

import torchvision.models as models  # 使用python版本<=python3.7，否则会有版本冲突
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from imblearn.metrics import sensitivity_score, specificity_score, geometric_mean_score
from sklearn.metrics import accuracy_score,  roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import *
import tensorflow.keras
import numpy as np
from sklearn import metrics
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print('torch.cuda.device_count()=', torch.cuda.device_count())
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)


def validate(batch_size, model, criterion,
             y_labels_val,
             data_set_mouthRIO_test,
             data_set_eyeLeftRIO_test, data_set_eyeRightRIO_test,
             data_set_location_test,
             data_set_iPPG_test,
             ):
    model.eval()
    total_loss_test = 0.0
    accuracy_test = 0
    y_true_test = []
    y_pred_test = []
    y_pred_probability_test = []
    y_pred_probability_arr_test = []
    with torch.no_grad():
        total_step = len(data_set_mouthRIO_test) // batch_size

        for j in range(total_step):
            if j < total_step - 1:
                # print('#################')
                # print('y_labels_val.shape = ',y_labels_val.shape)
                target = torch.from_numpy(y_labels_val[j * batch_size:(j + 1) * batch_size]).cuda()
                input_shape_mouth_test = torch.from_numpy(data_set_mouthRIO_test[j * batch_size:(j + 1) * batch_size, :]).cuda().float()
                input_shape_eyeLeft_test = torch.from_numpy(data_set_eyeLeftRIO_test[j * batch_size:(j + 1) * batch_size, :]).cuda().float()
                input_shape_eyeRight_test = torch.from_numpy(data_set_eyeRightRIO_test[j * batch_size:(j + 1) * batch_size, :]).cuda().float()
                input_shape_location_test = torch.from_numpy(data_set_location_test[j * batch_size:(j + 1) * batch_size, :]).cuda().float()
                input_shape_iPPG_test = torch.from_numpy(data_set_iPPG_test[j * batch_size:(j + 1) * batch_size, :]).cuda().float()
            else:
                target = torch.from_numpy(y_labels_val[j * batch_size:-1]).cuda()
                input_shape_mouth_test = torch.from_numpy(data_set_mouthRIO_test[j * batch_size:-1]).cuda().float()
                input_shape_eyeLeft_test = torch.from_numpy(data_set_eyeLeftRIO_test[j * batch_size:-1]).cuda().float()
                input_shape_eyeRight_test = torch.from_numpy(data_set_eyeRightRIO_test[j * batch_size:-1]).cuda().float()
                input_shape_location_test = torch.from_numpy(data_set_location_test[j * batch_size:-1]).cuda().float()
                input_shape_iPPG_test = torch.from_numpy(data_set_iPPG_test[j * batch_size:-1]).cuda().float()

            out_test = model(
                input_shape_mouth_test,
                input_shape_eyeLeft_test, input_shape_eyeRight_test,
                input_shape_location_test,
                input_shape_iPPG_test,
                # input_shape_eTou_test,
            )
            loss_test = F.cross_entropy(out_test.cuda().float(), target.cuda())
            # loss = F.cross_entropy(out, target)
            total_loss_test += loss_test.item()
            # accuracy += (torch.argmax(out_test, dim=1) == torch.tensor(target, dtype=torch.long)).sum().item()
            accuracy_test += (torch.argmax(out_test.cuda(), dim=1) == target.cuda()).sum().item()
            y_true_test.extend(target.cpu().numpy())
            y_pred_test.extend(torch.argmax(out_test, dim=1).cpu().numpy())
            y_pred_probability_test.extend(F.sigmoid(out_test).cpu().detach().numpy())  # 计算softmax，即属于各类的概率

    y_pred_probability_test = np.array(y_pred_probability_test)
    y_pred_probability_test = y_pred_probability_test[:, 1]
    acc = accuracy_score(y_true_test, y_pred_test)
    print('acc = ', acc)
    return y_pred_test, y_true_test, y_pred_probability_test, acc

class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing.
            # May need to write a CUDA kernel.
            raise NotImplementedError
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)


class InplaceShift(torch.autograd.Function):
    # Special thanks to @raoyongming for the help to this function
    @staticmethod
    def forward(ctx, input, fold):
        # not support higher order gradient
        # input = input.detach_()
        ctx.fold_ = fold
        n, t, c, h, w = input.size()
        buffer = input.data.new(n, t, fold, h, w).zero_()
        buffer[:, :-1] = input.data[:, 1:, :fold]
        input.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, 1:] = input.data[:, :-1, fold: 2 * fold]
        input.data[:, :, fold: 2 * fold] = buffer
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.detach_()
        fold = ctx.fold_
        n, t, c, h, w = grad_output.size()
        buffer = grad_output.data.new(n, t, fold, h, w).zero_()
        buffer[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
        grad_output.data[:, :, fold: 2 * fold] = buffer
        return grad_output, None


class TemporalPool(nn.Module):
    def __init__(self, net, n_segment):
        super(TemporalPool, self).__init__()
        self.net = net
        self.n_segment = n_segment

    def forward(self, x):
        x = self.temporal_pool(x, n_segment=self.n_segment)
        return self.net(x)

    @staticmethod
    def temporal_pool(x, n_segment):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        x = F.max_pool3d(x, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        x = x.transpose(1, 2).contiguous().view(nt // 2, c, h, w)
        return x


def make_temporal_shift(net, n_segment, n_div=8, place='blockres', temporal_pool=False):
    if temporal_pool:
        n_segment_list = [n_segment, n_segment // 2, n_segment // 2, n_segment // 2]
    else:
        n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0
    print('=> n_segment per stage: {}'.format(n_segment_list))

    import torchvision
    if isinstance(net, torchvision.models.ResNet):
        if place == 'block':
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    blocks[i] = TemporalShift(b, n_segment=this_segment, n_div=n_div)
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

        elif 'blockres' in place:
            n_round = 1
            if len(list(net.layer3.children())) >= 23:
                n_round = 2
                print('=> Using n_round {} to insert temporal shift'.format(n_round))

            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks residual'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    if i % n_round == 0:
                        blocks[i].conv1 = TemporalShift(b.conv1, n_segment=this_segment, n_div=n_div)
                return nn.Sequential(*blocks)

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])
    else:
        raise NotImplementedError(place)


def make_temporal_pool(net, n_segment):
    import torchvision
    if isinstance(net, torchvision.models.ResNet):
        print('=> Injecting nonlocal pooling')
        net.layer2 = TemporalPool(net.layer2, n_segment)
    else:
        raise NotImplementedError


class resnet50_TSM(nn.Module):
    def __init__(self, num_classes=2):
        super(resnet50_TSM, self).__init__()
        # pretrained=True 加载网络结构和预训练参数，False 时代表只加载网络结构，不加载预训练参数，即不需要用预训练模型的参数来初始化
        self.model = models.resnet50(pretrained=True)                                   # 调用模型
        fc_features = self.model.fc.in_features                                         # 提取 fc 层中固定的参数 in_features
        self.model.fc = nn.Linear(in_features=fc_features, out_features=128)    # 修改 fc 层中 out_features 参数，修改分类为9
        self.img_num = 380 * 8
        self.fc_picture = nn.Sequential(nn.Linear(self.img_num * 128, 128))
        ## LSTM
        self.n_units = 128
        self.n_inputs = 468  # 3DCNN输出的特征大小.shape[2] = 200
        self.rnn = nn.LSTM(
            input_size = self.n_inputs,
            hidden_size = self.n_units,
            num_layers = 1,
            batch_first = True,
        )
        ###  ippg
        self.n_inputs_iPPG = 18
        self.rnn_iPPG = nn.LSTM(
            input_size=self.n_inputs_iPPG,
            hidden_size=self.n_units,
            num_layers=1,
            batch_first=True,
        )
        ##TSM
        self.tsm = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=False)

        ###############################
        num_features = 5  # 根据输入特征的个数进行修改
        self.fc_sum = nn.Sequential(
            nn.Linear(num_features * 128, num_classes),
            nn.Sigmoid(),  # 和nn.BCELoss()配合使用，详见https://blog.csdn.net/weixin_40522801/article/details/1
            )

    def feature_extract_LSTM(self, x):
        # print('x.shape= ', x.shape)
        # print('type(x) = ', type(x))
        self.rnn = self.rnn.cuda()
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = r_out[:, -1, :]
        return out

    def feature_extract(self, x):
        features_arr = []
        self.model = self.model.cuda()
        # print('x.shape = ', x.shape) # [2, 100, 80, 80, 3]

        for i in range(x.shape[0]):
            output = x[i]  # [100, 80, 80, 3]
            output = x[i].transpose(1,3)   #  数据维度变换 # [100, 3, 80, 80]
            input_tsm = output[0:self.img_num, :, :, :]  # [self.img_num , 3, 80, 80]
            output = self.tsm(input_tsm)   # TSM 处理 ——input_tsm：(N* 8, C=3, W, H=W)   # [96, 3, 80, 80]
            output = self.model(output)    # 特征处理
            # 降维处理
            output = output.view(output.size(0), -1)
            output = torch.reshape(output, (-1,)) # [96*128]
            features_arr.append(output)

        features_arr = torch.stack(features_arr)
        features_arr = features_arr.cpu().detach().numpy()
        features_arr = self.fc_picture(torch.from_numpy(features_arr))
        return features_arr

    def feature_extract_iPPG(self, x):
        # print('x.shape= ', x.shape)
        self.rnn_iPPG = self.rnn_iPPG.cuda()
        r_out, (h_n, h_c) = self.rnn_iPPG(x, None)
        # out = self.out(r_out[:, -1, :])
        out = r_out[:, -1, :]
        return out

    def forward(self,
                input_shape_mouth,
                input_shape_eyeLeft, input_shape_eyeRight,
                input_shape_xyz,
                input_shape_iPPG,
                ):

        features_mouth = self.feature_extract(input_shape_mouth)
        features_eyeLeft = self.feature_extract(input_shape_eyeLeft)
        features_eyeRight = self.feature_extract(input_shape_eyeRight)
        features_xyz = self.feature_extract_LSTM(input_shape_xyz)
        features_iPPG = self.feature_extract_iPPG(input_shape_iPPG)
        # features_eTou = self.feature_extract(input_shape_eTou)
        # print('input_shape_xyz.shape = ', input_shape_xyz.shape)
        # print('features_xyz.shape = ', features_xyz.shape)
        # print('features_eyeRight.shape = ', features_eyeRight.shape)

        out = torch.cat((
                features_mouth.cuda(),
                features_eyeLeft.cuda(), features_eyeRight.cuda(),
                features_xyz.cuda(),
                features_iPPG,
            ), 1)  # 在 1 维(横向)进行拼接
        # out = features_mouth
        self.fc_sum = self.fc_sum.cuda()
        out = self.fc_sum(out)
        # x = self.model(x)
        return out

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

def data_pre(x_data, y_data, z_data): # (56, 6090, 468)
    loc_arr = []
    for i in range(x_data.shape[0]):
        for j in range(x_data.shape[1]):
            for m in range(x_data.shape[2]):
                distance = x_data[i,j,m]*x_data[i,j,m] + y_data[i,j,m]*y_data[i,j,m] + z_data[i,j,m]*z_data[i,j,m]
                loc_arr.append(distance)
    loc_arr = np.reshape(loc_arr, (x_data.shape[0],x_data.shape[1],x_data.shape[2]))
    return loc_arr

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
    data_arr = np.reshape(data_arr,(data.shape[0], int(data.shape[1]/2),data.shape[2],data.shape[3],data.shape[4]))
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
    for i in range(num_volunteer): # label: stress-free
        label.append(0)
    for i in range(num_volunteer):  # label: stress
        label.append(1)
    label = np.array(label)
    print('label.shape = ', label.shape) # (217,)
    # 一、 行为数据：眼睛、嘴巴、头部位姿
    # 1. 嘴巴
    data_set_mouthRIO_T1 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/mouth_ROI_T1_56_6090_40_40pixels/'
        + 'mouth_ROI_T1_56_6090_10_10pixels.npy') / 255
    data_set_mouthRIO_T2 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/mouth_ROI_T2_56_6090_40_40pixels/'
        + 'mouth_ROI_T2_56_6090_10_10pixels.npy') / 255
    data_set_mouthRIO_T1 = img_dowm_sample(data_set_mouthRIO_T1)
    data_set_mouthRIO_T2 = img_dowm_sample(data_set_mouthRIO_T2)
    data_set_mouthRIO = np.concatenate((data_set_mouthRIO_T1, data_set_mouthRIO_T2), axis=0)
    print('data_set_mouthRIO_T1.shape = ', data_set_mouthRIO_T1.shape)
    print('data_set_mouthRIO_T2.shape = ', data_set_mouthRIO_T2.shape)
    print('data_set_mouthRIO.shape = ', data_set_mouthRIO.shape)
    del data_set_mouthRIO_T1, data_set_mouthRIO_T2

    # 2. 左眼
    data_set_eyeLeftRIO_T1 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/eyeLeft_ROI_T1_56_6090_40_40pixels/'
        + 'eyeLeft_ROI_T1_56_6090_10_10pixels.npy') / 255
    data_set_eyeLeftRIO_T2 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/eyeLeft_ROI_T2_56_6090_40_40pixels/'
        + 'eyeLeft_ROI_T2_56_6090_10_10pixels.npy') / 255
    data_set_eyeLeftRIO_T1 = img_dowm_sample(data_set_eyeLeftRIO_T1)
    data_set_eyeLeftRIO_T2 = img_dowm_sample(data_set_eyeLeftRIO_T2)
    data_set_eyeLeftRIO = np.concatenate((data_set_eyeLeftRIO_T1, data_set_eyeLeftRIO_T2), axis=0)
    print('data_set_eyeLeftRIO_T1.shape = ', data_set_eyeLeftRIO_T1.shape)
    print('data_set_eyeLeftRIO_T2.shape = ', data_set_eyeLeftRIO_T2.shape)
    print('data_set_eyeLeftRIO.shape = ', data_set_eyeLeftRIO.shape)
    del data_set_eyeLeftRIO_T1, data_set_eyeLeftRIO_T2

    # 3. 右眼
    data_set_eyeRightRIO_T1 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/eyeRight_ROI_T1_56_6090_40_40pixels/'
        + 'eyeRight_ROI_T1_56_6090_10_10pixels.npy') / 255
    data_set_eyeRightRIO_T2 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/eyeRight_ROI_T2_56_6090_40_40pixels/'
        + 'eyeRight_ROI_T2_56_6090_10_10pixels.npy') / 255
    data_set_eyeRightRIO_T1 = img_dowm_sample(data_set_eyeRightRIO_T1)
    data_set_eyeRightRIO_T2 = img_dowm_sample(data_set_eyeRightRIO_T2)
    data_set_eyeRightRIO = np.concatenate((data_set_eyeRightRIO_T1, data_set_eyeRightRIO_T2), axis=0)
    print('data_set_eyeRightRIO_T1.shape = ', data_set_eyeRightRIO_T1.shape)
    print('data_set_eyeRightRIO_T2.shape = ', data_set_eyeRightRIO_T2.shape)
    print('data_set_eyeRightRIO.shape = ', data_set_eyeRightRIO.shape)
    del data_set_eyeRightRIO_T1, data_set_eyeRightRIO_T2

   

    # 4.头部位姿：
    loc_arr_T1 = np.load( '/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T1/loc_T1_56_6090_468/'
                          +'loc_distan_T1_56_6090_468.npy')
    loc_arr_T2 = np.load(
        '/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/loc_T2_56_6090_468/'
        + 'loc_distan_T2_56_6090_468.npy')
    print('loc_arr_T1.shape = ', loc_arr_T1.shape) # (56, 6090, 468)
    print('loc_arr_T2.shape = ', loc_arr_T2.shape) # (56, 6090, 468)
    loc = np.concatenate((loc_arr_T1, loc_arr_T2), axis=0)
    print('loc.shape = ', loc.shape)  # (112, 6090, 468)
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
    y_labels_train, y_labels_val = train_test_split(label, test_size=test_size,
                                                    random_state=random_state)
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
    model = resnet50_TSM(num_classes=2)
    criterion = nn.BCELoss()  # https://blog.csdn.net/weixin_40522801/article/details/106616564
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print('len(y_labels_train)  = ', len(y_labels_train))
    total_step = len(y_labels_train) // batch_size
    curr_lr = learning_rate
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
                labels = torch.from_numpy(y_labels_train[i * batch_size:(i + 1) * batch_size]).cuda()
                input_shape_mouth = torch.from_numpy(data_set_mouthRIO_train[i * batch_size:(i + 1) * batch_size, :]).cuda().float()
                input_shape_eyeLeft = torch.from_numpy(data_set_eyeLeftRIO_train[i * batch_size:(i + 1) * batch_size, :]).cuda().float()
                input_shape_eyeRight = torch.from_numpy(data_set_eyeRightRIO_train[i * batch_size:(i + 1) * batch_size, :]).cuda().float()
                input_shape_location = torch.from_numpy(data_set_location_train[i * batch_size:(i + 1) * batch_size, :]).cuda().float()
                input_shape_iPPG = torch.from_numpy(data_set_iPPG_train[i * batch_size:(i + 1) * batch_size]).cuda().float()

            else:
                labels = torch.from_numpy(y_labels_train[i * batch_size:-1]).cuda()
                input_shape_mouth = torch.from_numpy(data_set_mouthRIO_train[i * batch_size:-1]).cuda().float()
                input_shape_eyeLeft = torch.from_numpy(data_set_eyeLeftRIO_train[i * batch_size:-1]).cuda().float()
                input_shape_eyeRight = torch.from_numpy(data_set_eyeRightRIO_train[i * batch_size:-1]).cuda().float()
                input_shape_location = torch.from_numpy(data_set_location_train[i * batch_size:-1]).cuda().float()
                input_shape_iPPG = torch.from_numpy(data_set_iPPG_train[i * batch_size:-1]).cuda().float()
            # Forward pass
            optimizer.zero_grad()
            model.train()
            with torch.no_grad():
                outputs = model(
                                input_shape_mouth,
                                input_shape_eyeLeft, input_shape_eyeRight,
                                input_shape_location,
                                input_shape_iPPG,
                                # input_shape_forehead,
                                )
                # loss = F.cross_entropy(outputs, torch.tensor(labels, dtype=torch.long))
                # loss = F.cross_entropy(outputs, torch.tensor(labels, dtype=torch.long))
                # print("type(outputs) =", type(outputs))
                # print("type(labels) =", type(labels))
                loss = F.cross_entropy(outputs.cuda().float(), labels.cuda())
                loss = loss.requires_grad_()
                loss.backward()
                optimizer.step()
                if (i + 1) % 34 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}'
                          .format(epoch + 1, no_epochs, i + 1, total_step, loss.item()))
                    current_losses.append(loss.item())  # appends the current value of the loss into a list
                    ## 评价模型
                    y_pred_test, y_true_test, y_pred_probability_test, acc = validate(batch_size, model, criterion,
                             y_labels_val,
                             data_set_mouthRIO_test,
                             data_set_eyeLeftRIO_test, data_set_eyeRightRIO_test,
                             data_set_location_test,
                             data_set_iPPG_test,
                             )
                    if acc > temp_acc:
                        temp_acc = acc

                        y_test = y_true_test
                        preds = y_pred_test
                        test_predict = y_pred_probability_test
                        acc = accuracy_score(y_test, preds)
                        print('y_test = ', y_test)
                        print('test_predict = ', test_predict)
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
                        save_name = '_' + 'epochs' + str(no_epochs) + '_' + 'mouth'+ '_' + 'eyeLeft' + '_' + 'eyeRight'+ '_' + 'location'+ '_' + 'iPPG'
                        # + '_' + 'mouth' \
                        # + '_' + 'eyeLeft' + '_' + 'eyeRight' \
                        # + '_' + 'location'\
                        # + '_' + 'forehead'\
                        # + '_' + 'iPPG' # 额头提取iPPG
                        data_result = []
                        data_result.append([acc, roc_auc, precision, recall1, sensitivity, fscore1, specificity])
                        np.savetxt('./data_result_resnet50_TSM_10p_10pixel/resnet_3D_fpr' + save_name + '.txt', fpr)
                        np.savetxt('./data_result_resnet50_TSM_10p_10pixel/resnet_3D_tpr' + save_name + '.txt', tpr)
                        np.savetxt('./data_result_resnet50_TSM_10p_10pixel/resnet50_TSM_data_result' + save_name + '.txt', data_result)
                        np.savetxt('./data_result_resnet50_TSM_10p_10pixel/resnet50_TSM_y_test' + save_name + '.txt', y_true)
                        np.savetxt('./data_result_resnet50_TSM_10p_10pixel/resnet50_TSM_y_pred_probability' + save_name + '.txt',
                                   y_pred_probability_test)
                        np.savetxt('./data_result_resnet50_TSM_10p_10pixel/resnet50_TSM_y_pred' + save_name + '.txt', y_pred)
                        # 保存模型
                        # torch.save(model,
                        #            './data_result_resnet50_TSM_10p_10pixel/resnet50_TSM_mod4_0_1_anxiety_screeen' + save_name + '.pkl')


if __name__ == '__main__':
    train()




