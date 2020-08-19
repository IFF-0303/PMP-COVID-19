import os
import random
import imageio
import torch
import torch.nn as nn
import torch.functional as F
import torchvision.transforms.functional as TF
import numpy as np

from torch.autograd import Function
from resnet import generate_model
from torchvision import models
from densenet_3d import DenseNet
from torchvision import transforms as tfs

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def plot_scan(scan,i):
    temp = '{}.png'.format(i+1)
    imageio.imwrite(os.path.join('./temp/', temp), scan)

def plot_CT_scans(data):
    # data shape : (z, y, x)
    scans_len = data.shape[0]
    for i in range(scans_len):
        plot_scan(data[i],i)

def minmaxscaler(data):
    min = data.min()
    max = data.max()
    return (data - min)/(max-min+1e-6)


drop_rate = 0.2
drop_rate2 = 0.5 
NUM_CLASSES = 2
CHANNEL = 1


class Compute_class_score(nn.Module):
    def __init__(self,tau=3):
        super(Compute_class_score, self).__init__()
        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1.data.fill_(tau)

    def forward(self, class0_avg_vector, class1_avg_vector, data_vector):

        class_scores = torch.zeros(2, data_vector.shape[0]).cuda()
        class0_simi = torch.exp(self.w1*torch.cosine_similarity(data_vector, class0_avg_vector, dim=1))
        class1_simi = torch.exp(self.w1*torch.cosine_similarity(data_vector, class1_avg_vector, dim=1))
        class_scores[0] = torch.div(class0_simi, (class0_simi+class1_simi))
        class_scores[1] = 1.0-class_scores[0]
        
        return class_scores.permute(1,0), float(self.w1)



class Baseline(nn.Module):
    def __init__(self, num_classes=2):
        super(Baseline, self).__init__()  
        
        #16x16X16X80 => 16x16x16x64
        self.conv1 = nn.Sequential(
            nn.Conv3d(CHANNEL, 64, kernel_size=3, padding=1),
            nn.ReLU())
        
        #16x16x16x64 => 8x8x8x64
        self.conv15 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1,stride=2),
            nn.ReLU())
        
        #8x8x8x64 => 8x8x8x128
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.Dropout3d(p =drop_rate),
            nn.ReLU())
        
        #8x8x8x128 => 4x4x4x128
        self.conv25 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=1,stride=2),
            nn.BatchNorm3d(128),
            nn.Dropout3d(p =drop_rate),
            nn.ReLU())
        
        #4x4x4x128 => 4x4x4x256
        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.Dropout3d(p =drop_rate),
            nn.ReLU())
        
        #4x4x4x256 => 2x2x2x256
        self.conv35 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=3, padding=1,stride = 2),
            nn.BatchNorm3d(256),
            nn.Dropout3d(p =drop_rate),
            nn.ReLU())
        
        #2x2x2x256 => 2x2x2x512
        self.conv4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.Dropout3d(p =drop_rate),
            nn.ReLU())
        
        #2x2x2x512 => 1x1x1x512
        self.conv45 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1,stride=2),
            nn.BatchNorm3d(512),
            nn.Dropout3d(p =drop_rate),
            nn.ReLU())
        
        #1x1x1x512
        self.conv5 = nn.Sequential(
            nn.Conv3d(512, 128, kernel_size=1,stride=1),
            nn.BatchNorm3d(128),
            nn.Dropout3d(p =drop_rate2),
            nn.ReLU())
       
        # self.conv6 = nn.Sequential(
        #     nn.Conv3d(256, NUM_CLASSES, kernel_size=1,stride=1),
        #     nn.BatchNorm3d(NUM_CLASSES),
        #     nn.ReLU()) 
        
        self._convolutions = nn.Sequential(
            self.conv1,
            self.conv15,
            self.conv2,
            self.conv25,
            self.conv3,
            self.conv35,
            self.conv4,
            self.conv45,            
            self.conv5)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        out=self.conv1(x)
        out=self.conv15(out)
        out=self.conv2(out)
        out=self.conv25(out)
        out=self.conv3(out)
        out=self.conv35(out)
        out=self.conv4(out)
        out=self.conv45(out)            
        out=self.conv5(out)
        out = self.avg_pool(out)
        return out

class MedicNet(nn.Module):
    def __init__(self, config):
        
        super(MedicNet, self).__init__()
        self.bias           = config.fc_bias
        self.mlp_indim      = config.mlp_indim
        self.lstm_indim     = config.lstm_indim
        self.hidden_dim     = config.hidden_dim
        self.num_classes    = config.num_classes
        self.seq_len        = config.seq_len
        self.batch_size     = config.batch_size
        self.densenet_drop_rate = config.densenet_drop_rate
        self.encoder_3d     = config.encoder_3d
        self.sdata_pool = config.sdata_pool
        self.clinical_att = config.clinical_att
        self.att_weights = torch.ones((1,61)).cuda()*0.5
        self.clinical_backbone = config.clinical_backbone
        self.lstm_all_output = config.lstm_all_output
        self.lstm_att = config.lstm_att
        self.clinical_augmentation = config.clinical_augmentation   

        self.layer1 = self._make_layer( 2, 32, self.mlp_indim, 31, 1, stride=1)
        self.layer2 = self._make_layer( 32, 64, 31, 16, 1, stride=1)
        self.layer3 = self._make_layer( 64, 128, 16, 8, 1, stride=1)            
        

        self.sdata_encoder = nn.Sequential(
                        nn.Linear(self.mlp_indim, 64, bias=self.bias),

                        nn.LayerNorm(64),
                        nn.RReLU(inplace=True),
                
                        nn.Linear(64, 96, bias=self.bias),
                        nn.LayerNorm(96),
                        nn.RReLU(inplace=True),

                        nn.Linear(96, 128, bias=self.bias)
                        )

        self.clinical_res_bn_relu1 = nn.Sequential(
                        nn.BatchNorm1d(self.mlp_indim),
                        nn.ReLU(inplace=True),
                        )
        self.clinical_res_bn_relu2 = nn.Sequential(
                        nn.BatchNorm1d(self.mlp_indim),
                        nn.ReLU(inplace=True),
                        )
        self.clinical_res_bn_relu3 = nn.Sequential(
                        nn.BatchNorm1d(self.mlp_indim),
                        nn.ReLU(inplace=True),
                        )

        self.clinical_res_bn1 = nn.BatchNorm1d(self.mlp_indim)


        self.bn1 = nn.BatchNorm1d(self.mlp_indim)
        self.bn2 = nn.BatchNorm1d(self.mlp_indim)
        self.bn3 = nn.BatchNorm1d(self.mlp_indim)

        self.clinical_encoder_stage1 = nn.Sequential(
                        nn.Linear(self.mlp_indim, self.mlp_indim, bias=self.bias),
                        nn.ReLU(inplace=True)
                        )

        self.clinical_encoder_stage2 = nn.Sequential(
                        nn.Linear(self.mlp_indim, self.mlp_indim, bias=self.bias),
                        nn.ReLU(inplace=True)
                        )

        self.clinical_encoder_stage3 = nn.Sequential(
                        nn.Linear(self.mlp_indim, self.mlp_indim, bias=self.bias),
                        nn.ReLU(inplace=True)
                        )      

        self.classifier_final = nn.Sequential(
                        nn.Linear(config.hidden_dim*config.seq_len, 512, bias=self.bias),
                        nn.BatchNorm1d(512),
                        nn.ReLU(inplace=True),
                        nn.Linear(512, 256),
                        nn.BatchNorm1d(256),
                        nn.ReLU(inplace=True),
                        nn.Linear(256, config.num_classes, bias=self.bias),
                        )

        self.sdata_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.sdata_max_pool = nn.AdaptiveMaxPool1d(1)

        self.clinical_att = nn.Sequential(
                        nn.Linear(self.mlp_indim, 128),
                        nn.BatchNorm1d(128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, self.mlp_indim))

        self.lstm_att = nn.Sequential(
                        nn.Linear(self.lstm_indim, 128),
                        nn.BatchNorm1d(128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, self.lstm_indim))
      
        if config.seq_processor == 'lstm':
            self.lstm = nn.LSTM(self.lstm_indim, self.hidden_dim, num_layers=1)
        elif config.seq_processor == 'gru':
            self.lstm = nn.GRU(self.lstm_indim, self.hidden_dim, num_layers=1)
        else:
            raise NotImplementedError

        if config.encoder_3d == 'baseline':
            self.encoder_3d = Baseline(self.num_classes)
        elif config.encoder_3d == 'densenet':
            self.encoder_3d = DenseNet(bn_size=self.batch_size, drop_rate=self.densenet_drop_rate, config=config,efficient=True)
        elif config.encoder_3d == 'resnet':
            self.encoder_3d = generate_model(config)

        self.sdata_dropout = nn.Dropout(p = 0.02)

    def forward(self, x, alpha = 1.0):

        sdata = x['data'].float()
        batch_size = sdata.shape[0]

        if self.clinical_augmentation ==True:
            for i in range(batch_size):
                sdata[i] = self.sdata_dropout(sdata[i])

        if self.clinical_att==True:
            if self.training:
                # attention
                weights = self.clinical_att(sdata) 
                weights = weights.mean(dim=0).unsqueeze(0)
                weights = torch.sigmoid(weights)
                self.att_weights.detach()
                self.att_weights = self.att_weights * 0.95 + weights * 0.05
                sdata = weights * sdata

            else:
                self.att_weights.detach()
                sdata = self.att_weights * sdata
                print(self.att_weights)

        if self.clinical_backbone=='mlp+res':
            
            # stage1
            sdata = self.bn1(sdata)
            sdata_feat1 = self.clinical_encoder_stage1(sdata)         # mlp+bn+relu
            stage1_out = sdata_feat1 + sdata

            # stage2
            stage1_out = self.bn2(stage1_out)
            sdata_feat2 = self.clinical_encoder_stage2(stage1_out)
            stage2_out = sdata_feat2 + stage1_out

            # stage3
            stage2_out = self.bn3(stage2_out)   
            sdata_feat3 = self.clinical_encoder_stage3(stage2_out)
            sdata_feat = sdata_feat3 + stage2_out

        elif self.clinical_backbone=='resnet':

            sdata_feat = self.layer1(sdata)
            sdata_feat = self.layer2(sdata_feat)
            sdata_feat = self.layer3(sdata_feat)

            if self.sdata_pool=='avg':
                sdata_feat=self.sdata_avg_pool(sdata_feat).squeeze()
            elif self.sdata_pool=='max':
                sdata_feat=self.sdata_max_pool(sdata_feat).squeeze()

        if self.encoder_3d == None:
            output = self.classifier_sdata(sdata_feat)

        elif self.encoder_3d != None:
            embeding = torch.zeros(self.seq_len,batch_size,self.lstm_indim).cuda()
            ddata = x['CTs']

            for i in range(self.seq_len):
            
                scan = ddata[:,i,:,:,:].unsqueeze(1)
                embeding[i] = torch.cat((self.encoder_3d(scan),sdata_feat),axis=1)
                
            assert embeding.shape[0] == self.seq_len
            self.lstm.flatten_parameters()

            if self.lstm_att == True:
                lstm_weights = self.lstm_att(embeding) 
                lstm_weights = torch.sigmoid(lstm_weights)
                embeding = lstm_weights * embeding

            lstm_out,_ = self.lstm(embeding)
            lstm_out = lstm_out.permute(1,0,2)

            # Take the last one in the output sequence
            if self.lstm_all_output == False:
                lstm_out = lstm_out[:,-1,:]
          
            features = torch.reshape(lstm_out, (lstm_out.shape[0], -1))

            output = self.classifier_final(features)
            output = output.squeeze()
            if batch_size ==1:
                output = output.unsqueeze(0)

        return output, features

    def save(self, check_path, name):
        if os.path.exists(check_path):
            os.mkdir(check_path)
        torch.save(self.state_dict(), os.path.join(check_path, name))
        print(os.path.join(check_path, name) + '\t saved!')

    def _make_layer(self,  inchannel, outchannel, infeat_dim, outfeat_dim, block_num, stride=1):
        '''
        make layer, containing multiple Residual Blocks
        '''
        shortcut = nn.Sequential(
                nn.Conv1d(inchannel,outchannel,1,stride=2, bias=False),
                nn.BatchNorm1d(outchannel),
                nn.ReLU(inplace=True))
        
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, infeat_dim, outfeat_dim, stride, shortcut))
        
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel, outfeat_dim, outfeat_dim))
        return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
    '''
    sub-module: Residual Block
    '''
    def __init__(self, inchannel, outchannel, infeat_dim, outfeat_dim, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()

        self.left = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, 1, stride, 0, bias=False),
                nn.BatchNorm1d(outchannel),
                nn.ReLU(inplace=True),
            
                nn.Conv1d(outchannel, outchannel, 1, 1, 0, bias=False),
                nn.ReLU(inplace=True))

        self.right = shortcut
        self.fc_left = nn.Linear(infeat_dim, outfeat_dim, bias=False)
        self.bn_left = nn.BatchNorm1d(outchannel)
        self.drop_left = nn.Dropout(p=0.5)
                
        self.fc_right = nn.Linear(infeat_dim, outfeat_dim, bias=False)
        self.bn_right = nn.BatchNorm1d(outchannel)
        self.infeat_dim = infeat_dim
        self.outfeat_dim = outfeat_dim

    def forward(self, x):

        # left branch
        out = self.left(x)
        size = out.size()
        out = self.fc_left(out.view(-1,self.infeat_dim))
        out = self.drop_left(out)
        out = out.view(size[0],size[1],self.outfeat_dim)

        # right branch
        residual = x if self.right is None else self.right(x)

        out += residual
        return torch.relu(out)

class MultiFocalLoss(nn.Module):
    """
    Reference : https://www.zhihu.com/question/367708982/answer/985944528
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = torch.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss