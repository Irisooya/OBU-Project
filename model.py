import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as transforms
import random

####Data augmentation transformation function, used for input data####
# Define noise injection transforms
class GaussianNoise(object):
    def __init__(self, mean=0, std=0.1):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

class SaltPepperNoise(object):
    def __init__(self, prob=0.05):
        self.prob = prob
    def __call__(self, tensor):
        salt_pepper = torch.rand(tensor.size())
        tensor[salt_pepper < self.prob / 2] = 0.0
        tensor[salt_pepper > 1 - self.prob / 2] = 1.0
        return tensor

# Define color jitter transform
class RandomColorJitter(object):
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    def __call__(self, tensor):
        if random.random() > 0.5:
            tensor = transforms.functional.adjust_brightness(tensor, self.brightness)
            tensor = transforms.functional.adjust_contrast(tensor, 1 + self.contrast)
            tensor = transforms.functional.adjust_saturation(tensor, 1 + self.saturation)
            tensor = transforms.functional.adjust_hue(tensor, self.hue)
        return tensor

#####cas model and model for nerual network#############
####module for 1 classifier,1 convolutional layer#######
# ###Process the input feature data,generate classification results###
class CAS_Module(nn.Module):
    def __init__(self, len_feature, num_classes):
        super(CAS_Module, self).__init__()
        self.len_feature = len_feature
        #Feature extraction
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=2048, kernel_size=3,
                      stride=1, padding=1),#1
            nn.BatchNorm1d(2048),#2Batch normalization layer,rise precision
            nn.ReLU()#3
        )
        #class task
        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=num_classes, kernel_size=1,
                      stride=1, padding=0, bias=False)
        )
        self.drop_out = nn.Dropout(p=0.6)#Prevent overfitting
        # ######add more
        # self.conv2 = nn.Sequential(
        #     nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2)
        # )
        #在 __init__ 方法中,添加了一个 LSTM 层 rnn。
        # self.rnn = nn.LSTM(input_size=2048, hidden_size=2048, num_layers=1, batch_first=True)
####Initialize weights，improve effects
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        # input tensor x: (Batchsize, Timestep, Featurelength)B,T,F
        out = x.permute(0, 2, 1)
        # out: (B, F, T)
        out = self.conv(out)
        features = out.permute(0, 2, 1)
        #f change,turn before
        out = self.drop_out(out)
        #class f
        out = self.classifier(out)
        out = out.permute(0, 2, 1)
        # out: (B, T, C)
        return out, features

class Model(nn.Module):
    def __init__(self, len_feature, num_classes, r_act, r_bkg):
        super(Model, self).__init__()
        self.len_feature = len_feature
        self.num_classes = num_classes

        self.cas_module = CAS_Module(len_feature, num_classes)
#initial layers,classresult for deal with possibility
        self.softmax = nn.Softmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=2)

        self.r_act = r_act
        self.r_bkg = r_bkg
#######prevent overfitting######
        self.drop_out = nn.Dropout(p=0.5)

    def forward(self, x):
        num_segments = x.shape[1]#calculate number
        k_act = num_segments // self.r_act
        k_bkg = num_segments // self.r_bkg

        cas, features = self.cas_module(x)#

        feat_magnitudes = torch.norm(features, p=2, dim=2)
        select_idx = torch.ones_like(feat_magnitudes).cuda()
        select_idx = self.drop_out(select_idx)#random samplefind

        feat_magnitudes_drop = feat_magnitudes * select_idx
###calculate traverse value
        feat_magnitudes_rev = torch.max(feat_magnitudes, dim=1, keepdim=True)[0] - feat_magnitudes
        feat_magnitudes_rev_drop = feat_magnitudes_rev * select_idx
#feature index act,bkg
        _, sorted_idx = feat_magnitudes_drop.sort(descending=True, dim=1)
        idx_act = sorted_idx[:, :k_act]
        idx_act_feat = idx_act.unsqueeze(2).expand([-1, -1, features.shape[2]])
#extension
        _, sorted_idx = feat_magnitudes_rev_drop.sort(descending=True, dim=1)
        idx_bkg = sorted_idx[:, :k_bkg]
        idx_bkg_feat = idx_bkg.unsqueeze(2).expand([-1, -1, features.shape[2]])
        idx_bkg_cas = idx_bkg.unsqueeze(2).expand([-1, -1, cas.shape[2]])
#store
        feat_act = torch.gather(features, 1, idx_act_feat)
        feat_bkg = torch.gather(features, 1, idx_bkg_feat)

#def get_video_cls_scores(self, cas, k_act)
        sorted_scores, _= cas.sort(descending=True, dim=1)
        topk_scores = sorted_scores[:, :k_act, :]
        score_act = torch.mean(topk_scores, dim=1)
        score_bkg = torch.mean(torch.gather(cas, 1, idx_bkg_cas), dim=1)

        score_act = self.softmax(score_act)
        score_bkg = self.softmax(score_bkg)
#Probability value turn
        cas_softmax = self.softmax_2(cas)


        return score_act, score_bkg, feat_act, feat_bkg, features, cas_softmax

