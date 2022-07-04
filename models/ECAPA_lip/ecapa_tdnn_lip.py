from collections import OrderedDict
import os
from sqlite3 import Time
from turtle import distance, forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import glog as log
import torchvision

from utils.sample import Sample
from losses.angleproto import LOSS_DICT
from models.TalkNet.attentionLayer import attentionLayer
from models.ECAPA_pre.ecapa_tdnn_pre import ECAPA_TDNN
from models.ECAPA_lip.lip_model.model import get_lip_reading_model
from models.ECAPA_lip.resnet_facial_model import Resnet_facial

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

class Resnet18(nn.Module):
    def __init__(self, original_resnet, pool_type='maxpool', input_channel=3, with_fc=False, fc_in=512, fc_out=512):
        super(Resnet18, self).__init__()
        self.pool_type = pool_type
        self.input_channel = input_channel
        self.with_fc = with_fc

        #customize first convolution layer to handle different number of channels for images and spectrograms
        self.conv1 = nn.Conv2d(self.input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        layers = [self.conv1]
        layers.extend(list(original_resnet.children())[1:-2])
        self.feature_extraction = nn.Sequential(*layers) #features before pooling

        if with_fc:
            self.fc = nn.Linear(fc_in, fc_out)
            self.fc.apply(weights_init)

    def forward(self, x):
        x = self.feature_extraction(x)

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool2d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool2d(x, 1)
        else:
            return x

        if self.with_fc:
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
        else:
            return x

def get_visualvoice_facial_model(weight_pth):
    original_resnet = torchvision.models.resnet18(pretrained=False)
    net = Resnet18(original_resnet, pool_type='maxpool', with_fc=True, fc_in=512, fc_out=128)

    pretrained_state = torch.load(weight_pth, map_location=lambda storage, loc: storage)
    model_state = pretrained_state
    net.load_state_dict(model_state)
    return net

class AudioVideoReasoning(nn.Module):
    def __init__(self, feature_dim, audio_time_feature_dim, facial_feature_dim, lip_feature_dim, audio_emb_dim, facial_model_flag, config) -> None:
        super().__init__()
        self.facial_model_flag = facial_model_flag

        step_size = int(config.audio_num_frame*2/config.mouth_num_frame)
        assert step_size==config.audio_num_frame*2/config.mouth_num_frame
        self.audio_downsample_conv = nn.Conv1d(
            audio_time_feature_dim, feature_dim, step_size, stride=step_size, padding=0)

        self.mouth_linear = nn.Linear(lip_feature_dim, feature_dim)

        # Audio-visual Cross Attention
        self.crossA2V = attentionLayer(d_model = feature_dim, nhead = 8, positional_emb_flag=config.positional_emb_flag)
        self.crossV2A = attentionLayer(d_model = feature_dim, nhead = 8, positional_emb_flag=config.positional_emb_flag)

        # Audio-visual Self Attention
        self.selfAV = attentionLayer(d_model = feature_dim*2, nhead = 8, positional_emb_flag=config.positional_emb_flag)
        
        # visible token
        self.visible_token = nn.Embedding(2, feature_dim) # [2, 128]

        if facial_model_flag:
            self.facial_visible_token = nn.Embedding(2, facial_feature_dim) # [2, 512]
            self.concat_model = nn.Sequential(nn.Linear(feature_dim*2+facial_feature_dim, feature_dim*2),
                                            nn.ReLU(),
                                            nn.Linear(feature_dim*2, audio_emb_dim))
        elif feature_dim*2!=audio_emb_dim:
            self.output_linear = nn.Linear(feature_dim*2, audio_emb_dim)
        else:
            self.output_linear = nn.Identity()

    def forward_cross_attention(self, x1, x2):
        # [B*40, 50, 128], (B*num_way*num_shot, num_frame, 128)
        x1_c = self.crossA2V(src = x1, tar = x2)
        x2_c = self.crossV2A(src = x2, tar = x1)
        return x1_c, x2_c

    def forward_audio_visual_backend(self, x1, x2):
        x = torch.cat((x1,x2), 2) # [B, 64(T), feature_dim*2]
        x = self.selfAV(src = x, tar = x) # [B, 64(T), feature_dim*2]
        return x

    def forward(self, visible, audio_feature_time, visual_feature_time, facial_feature):
        # audio_feature_time [B*num_way*num_shot*num_segment, C, T] (B*20*1, 1536, 127)
        # visual_feature_time [B*num_way_shot*num_segment, C(512), T(64)]
        audio_feature_time = F.pad(audio_feature_time, (0,1), 'replicate') # (B*20*1, 1536, 128)
        audio_feature_time = self.audio_downsample_conv(audio_feature_time) # (B*20*1, 128 (C), 64 (T))
        audio_feature_time = audio_feature_time.transpose(1, 2) # [B*num_way*num_shot*num_segment, T, C] (B*20*1, 64, 128)
        
        visual_feature_time = self.mouth_linear(visual_feature_time.transpose(1, 2)) # [B*num_way_shot*num_segment, num_frame(64), C(128)]

        B_ws_s, num_frame, _ = visual_feature_time.shape
        visible = visible.to(visual_feature_time.device) # [B, 20] (B, num_way*num_shot)
        num_segment = int(B_ws_s/visible.shape[0]/visible.shape[1])
        token = self.visible_token(visible) # [B, 20, 128] (B, num_way*num_shot, D)
        token = token.unsqueeze(2).unsqueeze(3).expand(-1, -1, num_segment, num_frame, -1) # [B, 20, 1, 64, 128] (B, num_way*num_shot, num_segment, num_frame, D)
        token = token.reshape(B_ws_s, num_frame, visual_feature_time.shape[-1]) # [B*20*num_segment, 64, 128] (B*num_way*num_shot*num_segment, T, 128)
        visual_feature_time = visual_feature_time*token # [B*20*num_segment, 64, 128] (B*num_way*num_shot*num_segment, T, 128)

        audio_feature_time, visual_feature_time = self.forward_cross_attention(audio_feature_time, visual_feature_time) # [B*num_way*num_shot*num_segment, num_frame, channel(128)]
        audio_lip_feature = self.forward_audio_visual_backend(audio_feature_time, visual_feature_time)  # [B*num_way*num_shot*num_segment, num_frame, channel(128)*2]
        audio_lip_feature = torch.mean(audio_lip_feature, dim=1) # [B*num_way*num_shot*num_segment, channel(128)*2]
        if self.facial_model_flag:
            # [B*num_way_shot*num_segment*num_face, C (512)]
            facial_token = self.facial_visible_token(visible) # [B, 20, 512] (B, num_way*num_shot, D)
            facial_token = facial_token.unsqueeze(2).expand(-1, -1, num_segment, -1) # [B, 20, num_segment, 512] (B, num_way*num_shot, num_segment, D)
            facial_feature = facial_feature*(facial_token.reshape(B_ws_s, facial_feature.shape[-1]))
            AV_feature = self.concat_model(torch.cat([audio_lip_feature, facial_feature], dim=1)) # [B*num_way*num_shot*num_segment, channel(128)*2]
        else:
            AV_feature = self.output_linear(audio_lip_feature)
        return AV_feature

class ECAPA_lip_model(nn.Module):
    def __init__(self, config, num_way, num_shot):
        super().__init__()

        self.num_way = num_way
        self.num_shot = num_shot
        self.model_mode = config.model_mode
        self.test_normalize = config.test_normalize
        self.take_middle_feature = config.take_middle_feature
        self.distance = config.distance
        self.fix_audio = config.fix_audio
        self.fix_lip_reading = config.fix_lip_reading
        self.fix_facial = config.fix_facial
        self.facial_model_name = config.facial_model
        self.audio_feature_loaded = config.audio_feature_loaded
        self.mouth_feature_loaded = config.mouth_feature_loaded
        self.lip_identity_feature_dim = config.lip_identity_feature_dim
        
        self.audio_emb_dim = 256
        if not self.audio_feature_loaded:
            self.audio_feature_model = ECAPA_TDNN(feat_dim=768, channels=512, emb_dim=self.audio_emb_dim, feat_type='wavlm_base_plus',
                                                sr=16000, feature_selection="hidden_states", update_extract=False, config_path=None,
                                                torch_hub_pth=config.ECAPA_hub_pth)
            self.load_ECAPA_feature_model(config, time_pooling=True)
        
        if not self.mouth_feature_loaded:
            self.lip_reading_model = get_lip_reading_model(os.path.join(config.visualvoice_checkpoint_pth, 'lipreading_best.pth'))

        if config.lip_identity_feature_dim>0:
            assert config.lip_identity_feature_dim==128
            self.lip_identity_model = get_visualvoice_facial_model(os.path.join(config.visualvoice_checkpoint_pth, 'facial_best.pth'))

        if config.facial_model=='visualvoice':
            self.facial_model = get_visualvoice_facial_model(os.path.join(config.visualvoice_checkpoint_pth, 'facial_best.pth'))
            facial_feature_dim = 128
        elif config.facial_model=='resnet':
            self.facial_model = Resnet_facial(init_weight=config.resnet_facial_model_pth)
            facial_feature_dim = self.facial_model.num_features
        else:
            facial_feature_dim = 0

        if self.model_mode=='audiovideo':
            feature_dim = config.reason_feature_dim
            self.audioVideoReason_model = AudioVideoReasoning(feature_dim=feature_dim, 
                                                            audio_time_feature_dim=1536,
                                                            facial_feature_dim=facial_feature_dim,
                                                            lip_feature_dim=512+self.lip_identity_feature_dim,
                                                            audio_emb_dim=self.audio_emb_dim, 
                                                            facial_model_flag = not (not self.facial_model_name),
                                                            config=config)
        else:
            self.tmp = nn.Parameter(torch.tensor([2.0]))


        loss_cls = LOSS_DICT[config.loss_param.type]
        self.criterion = loss_cls(config.loss_param.param)

        self.fix_models()

    def state_dict(self):
        if self.model_mode=='audiovideo':
            return self.audioVideoReason_model.state_dict()
        else:
            return {}
    
    def load_state_dict(self, state_dict, strict=True):
        self.audioVideoReason_model.load_state_dict(state_dict, strict)

    def fix_models(self):
        if self.fix_audio and (not self.audio_feature_loaded):
            for param in self.audio_feature_model.parameters():
                param.requires_grad = False
        
        if self.fix_lip_reading and (not self.mouth_feature_loaded):
            for param in self.lip_reading_model.parameters():
                param.requires_grad = False

        if self.fix_lip_reading and self.lip_identity_feature_dim>0:
            for param in self.lip_identity_model.parameters():
                param.requires_grad = False

        if self.fix_facial and self.facial_model_name:
            for param in self.facial_model.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.fix_audio and (not self.audio_feature_loaded):
            for m in self.audio_feature_model.modules():
                m.eval()
        if self.fix_lip_reading and (not self.mouth_feature_loaded):
            for m in self.lip_reading_model.modules():
                m.eval()
        if self.fix_lip_reading and self.lip_identity_feature_dim>0:
            for m in self.lip_identity_model.modules():
                m.eval()
        if self.fix_facial and self.facial_model_name:
            for m in self.facial_model.modules():
                m.eval()

    def load_ECAPA_feature_model(self, config, time_pooling):
        if config.ECAPA_load_checkpoint:
            state_dict = torch.load(config.ECAPA_checkpoint, map_location=lambda storage, loc: storage)
            state_dict['model'].pop('loss_calculator.projection.weight')
            
            if not time_pooling:
                new_state_dict = OrderedDict()
                for k,v in state_dict['model'].items():
                    if k.startswith('pooling') or k.startswith('bn') or k.startswith('linear'):
                        pass
                    else:
                        new_state_dict[k] = v
                state_dict['model']=new_state_dict

            self.audio_feature_model.load_state_dict(state_dict['model'], strict=True)
            log.info('Load checkpoint for ECAPA_pre feature model from: %s'%config.ECAPA_checkpoint)
        else:
            log.info('Did not load checkpoint for ECAPA_pre feature model')

    def get_facial_feature(self, batch):
        if self.facial_model_name=='visualvoice':
            frames = batch.frames_visualvoice
        else:
            frames = batch.frames
        B, num_way_shot, num_segment, num_face, C, W, H = frames.shape
        assert num_face==1
        facial_feature = self.facial_model(frames.reshape(B*num_way_shot*num_segment, C, W, H))
                # [B*num_way_shot*num_segment*num_face, C (128)]

        if self.facial_model_name!='visualvoice':
            facial_feature = F.normalize(facial_feature, p=2, dim=1)
        return facial_feature

    def get_audio_feature(self, audio):
        # [B, 40, num_segment, 32000] (B, num_way*num_shot, num_segment, 32000)
        B, num_way_shot, num_segment, L = audio.shape
        audio = audio.reshape([B*num_way_shot*num_segment, L])
        audio_feature_pre, audio_feature_time = self.audio_feature_model(audio) # [B*num_way*num_shot*num_segment, C (256)]
                            # [B*num_way*num_shot*num_segment, C, T] (B*20*1, 1536, 99)
        if self.fix_audio:
            audio_feature_pre = audio_feature_pre.detach()
            audio_feature_time = audio_feature_time.detach()
        return audio_feature_pre, audio_feature_time

    def forward(self, batch, mode):
        output = Sample()

        if mode=='train' or mode=='extract':
            if 'audio_feature' in batch.keys():
                B, num_way_shot, num_segment, _ = batch.audio_feature.shape
                audio_feature_pre = batch.audio_feature.reshape([B*num_way_shot*num_segment, -1])
                audio_feature_time = batch.audio_feature_time.reshape([B*num_way_shot*num_segment, batch.audio_feature_time.shape[3], batch.audio_feature_time.shape[4]])
            else:
                B, num_way_shot, num_segment, _ = batch.audio.shape
                audio_feature_pre, audio_feature_time = self.get_audio_feature(batch.audio)

            if 'audio_original_feature' in batch.keys():
                audio_original_feature_pre = batch.audio_original_feature.reshape([B*num_way_shot*num_segment, -1])
            elif 'audio_original' in batch.keys():
                audio_original_feature_pre, _ = self.get_audio_feature(batch.audio_original)
            else:
                audio_original_feature_pre = audio_feature_pre

            if self.model_mode=='audio':
                AV_feature = audio_feature_pre

            elif self.model_mode=='audioface_avg':
                assert self.facial_model_name
                facial_feature = self.get_facial_feature(batch)
                facial_feature = facial_feature.detach()
                AV_feature = torch.cat([audio_feature_pre, facial_feature], dim=1)
                
            elif self.model_mode=='audiovideo':
                if self.mouth_feature_loaded:
                    B, num_way_shot, num_segment, C_lip, lip_1, num_frame = batch.mouth_feature_time.shape
                    assert lip_1==1
                    visual_feature_time = batch.mouth_feature_time.reshape([B*num_way_shot*num_segment, C_lip, 1, num_frame]) # [B*num_way_shot*num_segment, C(512), 1, num_frame(64)]
                else:
                    B, num_way_shot, num_segment, num_frame, C, W, H = batch.mouth_pics.shape
                    mouth_pics = batch.mouth_pics.reshape(B*num_way_shot*num_segment, num_frame, C, W, H).transpose(1, 2) # [B*num_way_shot*num_segment, C, num_frame, W, H]
                    visual_feature_time = self.lip_reading_model(mouth_pics) # [B*num_way_shot*num_segment, C(512), 1, num_frame(64)]
                    visual_feature_time = visual_feature_time.detach()

                if self.lip_identity_feature_dim>0:
                    B, num_way_shot, num_segment, num_face, C, W, H = batch.frames_visualvoice.shape
                    assert num_face==1
                    lip_identity_feature = self.lip_identity_model(batch.frames_visualvoice.reshape(B*num_way_shot*num_segment, C, W, H)) # [B*num_way_shot*num_segment, C(128)]
                    lip_identity_feature = lip_identity_feature.unsqueeze(2).unsqueeze(3).repeat(1, 1, 1, visual_feature_time.shape[3]) # [B*num_way_shot*num_segment, C(128), 1, num_frame]
                    visual_feature_time = torch.cat([lip_identity_feature, visual_feature_time], dim=1) # [B*num_way_shot*num_segment, C(128+512), 1, num_frame]

                if self.facial_model_name:
                    facial_feature = self.get_facial_feature(batch)
                else:
                    facial_feature=None

                AV_feature = self.audioVideoReason_model(visible=batch.visible, audio_feature_time=audio_feature_time,
                    visual_feature_time=visual_feature_time[:, :, 0, :], facial_feature=facial_feature) # [B*num_way_shot*num_segment, C(256)]
                
            AV_feature = AV_feature.reshape([B, num_way_shot, num_segment, AV_feature.shape[-1]]) # [B, num_way*num_shot(20), num_segment, C]

            if mode=='train':
                assert num_segment==1
                AV_feature = AV_feature.squeeze(2)
                AV_feature = AV_feature.reshape(B, self.num_way, self.num_shot, AV_feature.shape[-1]) # [B, num_way, num_shot, C]

                loss, prec1, loss_list =  self.criterion(AV_feature, batch.targets, audio_original_feature_pre, batch.loss_k)
                
                output.loss = loss
                output.prec1 = prec1
                output.loss_list = loss_list

            elif mode=='extract':
                assert num_way_shot==1
                output.av_feature = AV_feature # [B, num_way*num_shot, num_segment, channel(128)*2]

                if AV_feature.shape[-1]==audio_original_feature_pre.shape[-1]:
                    prec1 = self.criterion.test_forward(AV_feature, batch.targets, audio_original_feature_pre)
                    output.prec1 = prec1

        elif mode=='distance':
            av_feature_i = batch['av_feature_i'] # [B, num_segment, C]
            av_feature_j = batch['av_feature_j']
            visible_i = batch['visible_i'] # [B]
            visible_j = batch['visible_j']

            if self.model_mode=='audioface_avg':
                audio_distance = self.distance_between_feature(av_feature_i[:,:,:self.audio_emb_dim], av_feature_j[:,:,:self.audio_emb_dim])
                facial_distance = self.distance_between_feature(av_feature_i[:,:,self.audio_emb_dim:], av_feature_j[:,:,self.audio_emb_dim:])

                visible_batch = ((visible_i+visible_j)==2)
                facial_distance[visible_batch==0] = audio_distance[visible_batch==0]
                # facial_distance = audio_distance

                distance = (facial_distance+audio_distance)/2
            else:
                distance = self.distance_between_feature(av_feature_i, av_feature_j)

            output.scores = distance
        else:
            raise NotImplementedError()
        
        return output

    def distance_between_feature(self, feature_i, feature_j):
        # [B, num_segment, C]

        if self.test_normalize:
            feature_i = F.normalize(feature_i, p=2, dim=2)
            feature_j = F.normalize(feature_j, p=2, dim=2)

        num_segment = feature_i.shape[1]

        if self.take_middle_feature:
            feature_i = feature_i[:, num_segment//2:(num_segment//2+1), :] # [B, 1, C]
            feature_j = feature_j[:, num_segment//2:(num_segment//2+1), :]
            num_segment = 1
            
        feature_i_re = feature_i.unsqueeze(2).repeat(1, 1, num_segment, 1) # [B, num_segment, num_segment, C]
        feature_j_re = feature_j.unsqueeze(1).repeat(1, num_segment, 1, 1) # [B, num_segment, num_segment, C]
        if self.distance=='cosine':
            distance = F.cosine_similarity(feature_i_re, feature_j_re, dim=3) # [B, num_segment, num_segment]
            distance = (distance+1)/2.0 # 0~1
        elif self.distance=='L2':
            distance = F.pairwise_distance(feature_i_re, feature_j_re) # [B, num_segment, num_segment]
            distance = (2-distance)/2.0
        distance = distance.mean(2).mean(1) # [B]

        return distance
