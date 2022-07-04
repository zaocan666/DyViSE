#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VideoLoss(nn.Module):
    prec_name='mae'
    test_prec_name='cosine'
    loss_list_name=['mse_loss', 'feature_loss']
    def __init__(self, param):
        super(VideoLoss, self).__init__()

        self.feature_loss = param.feature_loss
        # self.feature_loss_k = param.feature_loss_k

        # print('Initialised AngleProto')

    def forward(self, x_in, label, audio_feature, loss_k):
        # x [B, num_way, num_shot (1), C]
        # label [B, num_way, num_shot]
        # audio_feature [B*num_way*num_shot*num_segment, C (256)]
        B, num_way, num_shot, C = x_in.shape
        assert num_shot == 1

        out_anchor = x_in.repeat(1, 1, num_way, 1) # [B, num_way, num_way, C]
        out_positive = out_anchor.transpose(1, 2) # [B, num_way, num_way, C]
        cos_sim_matrix  = F.cosine_similarity(out_anchor, out_positive, dim=3) # [B, num_way, num_way] -1~1

        label_matrix_il = label.repeat(1, 1, num_way) # [B, num_way, num_way]
        label_matrix_lj = label_matrix_il.transpose(1, 2) # [B, num_way, num_way]
        label_matrix = (label_matrix_il==label_matrix_lj).float() # [B, num_way, num_way], 1/0

        di = label_matrix.sum(2) # # [B, num_way]
        dij = torch.matmul(di.unsqueeze(2), di.unsqueeze(1))
        aim_matrix = label_matrix # [B, num_way, num_way], 1/0
        mse_loss = (torch.pow(cos_sim_matrix-aim_matrix, 2)/torch.sqrt(dij)).mean()
        
        prec1   = torch.abs(cos_sim_matrix.detach()-aim_matrix.detach()).mean().item()

        if self.feature_loss:
            feature_loss = get_feature_loss(x_in, audio_feature)
            nloss = loss_k*mse_loss + feature_loss
            loss_list = [mse_loss.item(), feature_loss.item()]
        else:
            nloss = mse_loss
            loss_list = [mse_loss.item(), 0]
        return nloss, prec1, loss_list

    def test_forward(self, x_in, label, audio_feature):
        feature_loss = get_feature_loss(x_in, audio_feature)
        prec1   = -feature_loss.item()
        return prec1

def get_feature_loss(x, feature):
    x = torch.reshape(x, [-1, x.shape[-1]])
    nloss = -F.cosine_similarity(x, feature, dim=1).mean()
    return nloss


LOSS_DICT = {
    'video': VideoLoss,
}