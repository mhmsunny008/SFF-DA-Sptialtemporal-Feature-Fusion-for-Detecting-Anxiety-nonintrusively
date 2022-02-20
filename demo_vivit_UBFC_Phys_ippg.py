import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module import Attention, PreNorm, FeedForward
import numpy as np
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
print('torch.cuda.device_count()=', torch.cuda.device_count())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


  
class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_frames, dim = 192, depth = 4, heads = 3, pool = 'cls', in_channels = 3, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, ):
        super().__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        mlp_head_output = 128
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_head_output)
        )


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

           ###############################
        num_features = 5  # 根据输入特征的个数进行修改
        self.fc_sum = nn.Sequential(
            nn.Linear(num_features * 128, num_classes),
            nn.Sigmoid(),  # 和nn.BCELoss()配合使用，详见https://blog.csdn.net/weixin_40522801/article/details/1
            )


    def feature_extract(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)
        

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)

    def feature_extract_LSTM(self, x):
        # print('x.shape= ', x.shape)
        # print('type(x) = ', type(x))
        self.rnn = self.rnn.cuda()
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = r_out[:, -1, :]
        return out

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
        out = torch.cat((
                features_mouth.cuda(),
                features_eyeLeft.cuda(), features_eyeRight.cuda(),
                features_xyz.cuda(),
                features_iPPG,
            ), 1)  # 在 1 维(横向)进行拼接
        # out = features_mouth
        self.fc_sum = self.fc_sum.cuda()
        out = self.fc_sum(out)

        return out
    
    
def binary_onehot(y_labels_train):
    y_labels_train_arr = []
    for i in range(len(y_labels_train)):
        if y_labels_train[i] != 0:
            y_labels_train_arr.append([1,0])
        else:
            y_labels_train_arr.append([0,1])
    return np.array(y_labels_train_arr)
def img_dowm_sample(data):
    # img_num_per_second = 3
    # img_num_per_second = 4
    img_num_per_second = 5
    data_arr = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if j%img_num_per_second==0:
                data_arr.append(data[i,j])
    data_arr = np.reshape(data_arr,(data.shape[0], int(data.shape[1]/img_num_per_second),data.shape[2],data.shape[3],data.shape[4]))
    del data
    return data_arr

def train():
    # train constants
    no_epochs = 50  # 5000 originally
    # learning_rate = 1e-5  # img
    # learning_rate = 1e-4  # img
    learning_rate = 1e-3  # ippg
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
        + 'mouth_ROI_T1_56_6090_40_40pixels.npy')/ 255
    data_set_mouthRIO_T2 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/mouth_ROI_T2_56_6090_40_40pixels/'
        + 'mouth_ROI_T2_56_6090_40_40pixels.npy')/ 255
    # data_set_mouthRIO_T1 = data_set_mouthRIO_T1[:, 0:img_num_tatal, :, :, :]
    # data_set_mouthRIO_T2 = data_set_mouthRIO_T2[:, 0:img_num_tatal, :, :, :]
    data_set_mouthRIO_T1 = img_dowm_sample(data_set_mouthRIO_T1)
    data_set_mouthRIO_T2 = img_dowm_sample(data_set_mouthRIO_T2)
    data_set_mouthRIO = np.concatenate((data_set_mouthRIO_T1, data_set_mouthRIO_T2), axis=0) #(56, 6090, 40, 40, 3)
    data_set_mouthRIO = data_set_mouthRIO.transpose((0, 1, 4, 2, 3))  # (112, 2030, 3, 40, 40)
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
    # data_set_eyeLeftRIO_T1 = data_set_eyeLeftRIO_T1[:, 0:img_num_tatal, :, :, :]
    # data_set_eyeLeftRIO_T2 = data_set_eyeLeftRIO_T2[:, 0:img_num_tatal, :, :, :]
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
        + 'eyeRight_ROI_T1_56_6090_40_40pixels.npy')/ 255
    data_set_eyeRightRIO_T2 = np.load(
        r'/home/ps/lab-data/MoHaiMiao/2021-05-anxiety_screen-paper2/2021-11-09-dataset_UBFC_Phys/dataset_UBFC_Phys_T2/eyeRight_ROI_T2_56_6090_40_40pixels/'
        + 'eyeRight_ROI_T2_56_6090_40_40pixels.npy')/ 255
    # data_set_eyeRightRIO_T1 = data_set_eyeRightRIO_T1[:, 0:img_num_tatal, :, :, :]
    # data_set_eyeRightRIO_T2 = data_set_eyeRightRIO_T2[:, 0:img_num_tatal, :, :, :]
    data_set_eyeRightRIO_T1 = img_dowm_sample(data_set_eyeRightRIO_T1)
    data_set_eyeRightRIO_T2 = img_dowm_sample(data_set_eyeRightRIO_T2)
    data_set_eyeRightRIO = np.concatenate((data_set_eyeRightRIO_T1, data_set_eyeRightRIO_T2), axis=0)
    data_set_eyeRightRIO = data_set_eyeRightRIO.transpose((0, 1, 4, 2, 3)) 
    print('data_set_eyeRightRIO_T1.shape = ', data_set_eyeRightRIO_T1.shape)
    print('data_set_eyeRightRIO_T2.shape = ', data_set_eyeRightRIO_T2.shape)
    print('data_set_eyeRightRIO.shape = ', data_set_eyeRightRIO.shape)
    del data_set_eyeRightRIO_T1, data_set_eyeRightRIO_T2


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
    # del data_set_mouthRIO, data_set_eyeLeftRIO, data_set_eyeRightRIO, loc, iPPG, data_set_ippg_forehead, data_set_ippg_nose

    # initiates model and loss
    no_classes = 2
    image_size = data_set_mouthRIO.shape[3]
    # batch_size = 2
    num_frames = data_set_mouthRIO.shape[1]
    del data_set_mouthRIO, data_set_eyeLeftRIO, data_set_eyeRightRIO, loc, iPPG, data_set_ippg_forehead, data_set_ippg_nose
    model = ViViT(image_size, batch_size, no_classes, num_frames).cuda()

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
                labels = torch.from_numpy(y_labels_train[i * batch_size:(i + 1) * batch_size]).cuda()
                input_shape_mouth= torch.from_numpy(data_set_mouthRIO_train[i * batch_size:(i + 1) * batch_size]).cuda().float()
                input_shape_eyeLeft = torch.from_numpy(data_set_eyeLeftRIO_train[i * batch_size:(i + 1) * batch_size]).cuda().float()
                input_shape_eyeRight = torch.from_numpy(data_set_eyeRightRIO_train[i * batch_size:(i + 1) * batch_size]).cuda().float()
                input_shape_xyz = torch.from_numpy(data_set_location_train[i * batch_size:(i + 1) * batch_size]).cuda().float()
                input_shape_iPPG = torch.from_numpy(data_set_iPPG_train[i * batch_size:(i + 1) * batch_size]).cuda().float()
            else:
                labels = torch.from_numpy(y_labels_train[i * batch_size:-1]).cuda()
                input_shape_mouth= torch.from_numpy(data_set_mouthRIO_train[i * batch_size:-1]).cuda().float()
                input_shape_eyeLeft = torch.from_numpy(data_set_eyeLeftRIO_train[i * batch_size:-1]).cuda().float()
                input_shape_eyeRight = torch.from_numpy(data_set_eyeRightRIO_train[i * batch_size:-1]).cuda().float()
                input_shape_xyz = torch.from_numpy(data_set_location_train[i * batch_size:-1]).cuda().float()
                input_shape_iPPG = torch.from_numpy(data_set_iPPG_train[i * batch_size:-1]).cuda().float()


            # Forward pass
            optimizer.zero_grad()
            model.train()
            # print('input_shape_mouth.shape', input_shape_mouth.shape)
            # print('input_shape_eyeLeft.shape', input_shape_eyeLeft.shape)
            # print('input_shape_eyeRight.shape', input_shape_eyeRight.shape)
            # print('input_shape_eTou.shape', input_shape_eTou.shape)
            with torch.no_grad():

                outputs = model(
                              input_shape_mouth,
                              input_shape_eyeLeft, input_shape_eyeRight,
                            input_shape_xyz,
                            input_shape_iPPG,
                            )
                # print('######')
                # print('outputs = ', outputs)
                # print('labels = ', labels )
                # print('######')
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
                    if acc >= temp_acc:
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
                        np.savetxt('./data_result_Vivit_40p_40pixel_UBFC_Phys/Vivit_3D_fpr' + save_name + '.txt', fpr)
                        np.savetxt('./data_result_Vivit_40p_40pixel_UBFC_Phys/Vivit_3D_tpr' + save_name + '.txt', tpr)
                        np.savetxt('./data_result_Vivit_40p_40pixel_UBFC_Phys/Vivit_data_result' + save_name + '.txt', data_result)
                        np.savetxt('./data_result_Vivit_40p_40pixel_UBFC_Phys/Vivit_y_test' + save_name + '.txt', y_true)
                        np.savetxt('./data_result_Vivit_40p_40pixel_UBFC_Phys/Vivit_y_pred_probability' + save_name + '.txt',
                                   y_pred_probability_test)
                        np.savetxt('./data_result_Vivit_40p_40pixel_UBFC_Phys/Vivit_y_pred' + save_name + '.txt', y_pred)
                        # 保存模型
                        # torch.save(model,
                        #            './data_result_resnet50_TSM_10p_10pixel/resnet50_TSM_mod4_0_1_anxiety_screeen' + save_name + '.pkl')


if __name__ == "__main__":
    train()

    # image_size = 40
    # patch_size = 2
    # num_classes  = 2
    # num_frames = 96
    # img = torch.ones([1, num_frames, 3, image_size, image_size]).cuda()
    
    # model = ViViT(image_size, patch_size, num_classes, num_frames).cuda()
    # # parameters = filter(lambda p: p.requires_grad, model.parameters())
    # # parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    # # print('Trainable Parameters: %.3fM' % parameters)
    
    # out = model(img)

    # print("Shape of out :", out.shape)      # [B, num_classes]

    
    