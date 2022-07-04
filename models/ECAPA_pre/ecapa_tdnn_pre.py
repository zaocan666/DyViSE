from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as trans
import glog as log

from utils.sample import Sample
from losses.angleproto import LOSS_DICT
from models.TalkNet.visualEncoder import visualFrontend, visualTCN, visualConv1D
from models.TalkNet.attentionLayer import attentionLayer

''' The SE connection of 1D case.
'''
class SE_Connect(nn.Module):
    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)

        return out


''' Res2Conv1d + BatchNorm1d + ReLU
'''
class Res2Conv1dReluBn(nn.Module):
    '''
    in_channels == out_channels == channels
    '''

    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, scale=4):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias))
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            # Order: conv -> relu -> bn
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)

        return out


class SE_Res2Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, scale, se_bottleneck_dim):
        super().__init__()
        self.Conv1dReluBn1 = Conv1dReluBn(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.Res2Conv1dReluBn = Res2Conv1dReluBn(out_channels, kernel_size, stride, padding, dilation, scale=scale)
        self.Conv1dReluBn2 = Conv1dReluBn(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.SE_Connect = SE_Connect(out_channels, se_bottleneck_dim)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x):
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = self.Conv1dReluBn1(x)
        x = self.Res2Conv1dReluBn(x)
        x = self.Conv1dReluBn2(x)
        x = self.SE_Connect(x)

        return x + residual


''' Conv1d + BatchNorm1d + ReLU
'''
class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


''' Attentive weighted mean and standard deviation pooling.
'''
class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, attention_channels=128, global_context_att=False):
        super().__init__()
        self.global_context_att = global_context_att

        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        if global_context_att:
            self.linear1 = nn.Conv1d(in_dim * 3, attention_channels, kernel_size=1)  # equals W and b in the paper
        else:
            self.linear1 = nn.Conv1d(in_dim, attention_channels, kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(attention_channels, in_dim, kernel_size=1)  # equals V and k in the paper

    def forward(self, x):
        # [B, C, L]
        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        alpha = torch.tanh(self.linear1(x_in))
        # alpha = F.relu(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class ECAPA_TDNN(nn.Module):
    def __init__(self, feat_dim=768, channels=512, emb_dim=256, global_context_att=False, 
                feat_type='wavlm_base_plus', sr=16000, feature_selection="hidden_states", 
                update_extract=False, config_path=None, time_pooling=True, torch_hub_pth=''):
        super().__init__()

        self.feat_type = feat_type
        self.feature_selection = feature_selection
        self.update_extract = update_extract
        self.sr = sr

        if feat_type == "fbank" or feat_type == "mfcc":
            self.update_extract = False

        win_len = int(sr * 0.025)
        hop_len = int(sr * 0.01)

        if feat_type == 'fbank':
            self.feature_extract = trans.MelSpectrogram(sample_rate=sr, n_fft=512, win_length=win_len,
                                                        hop_length=hop_len, f_min=0.0, f_max=sr // 2,
                                                        pad=0, n_mels=feat_dim)
        elif feat_type == 'mfcc':
            melkwargs = {
                'n_fft': 512,
                'win_length': win_len,
                'hop_length': hop_len,
                'f_min': 0.0,
                'f_max': sr // 2,
                'pad': 0
            }
            self.feature_extract = trans.MFCC(sample_rate=sr, n_mfcc=feat_dim, log_mels=False,
                                              melkwargs=melkwargs)
        else:
            if config_path is None:
                # self.feature_extract = torch.hub.load('s3prl/s3prl', feat_type)
                # self.feature_extract = torch.hub.load('/home/urkax/.cache/torch/hub/s3prl_s3prl_master/', feat_type, source='local', verbose=False)
                # self.feature_extract = torch.hub.load(
                #     '/home/trunk/RTrunk0/urkax/torch_hub/s3prl/s3prl_s3prl_master/', feat_type, source='local', verbose=False)
                self.feature_extract = torch.hub.load(torch_hub_pth, feat_type, source='local', verbose=False)

                self.feature_extract.eval()
            else:
                pass
                # self.feature_extract = UpstreamExpert(config_path)
            if len(self.feature_extract.model.encoder.layers) == 24 and hasattr(self.feature_extract.model.encoder.layers[23].self_attn, "fp32_attention"):
                self.feature_extract.model.encoder.layers[23].self_attn.fp32_attention = False
            if len(self.feature_extract.model.encoder.layers) == 24 and hasattr(self.feature_extract.model.encoder.layers[11].self_attn, "fp32_attention"):
                self.feature_extract.model.encoder.layers[11].self_attn.fp32_attention = False

            self.feat_num = self.get_feat_num()
            self.feature_weight = nn.Parameter(torch.zeros(self.feat_num))

        if feat_type != 'fbank' and feat_type != 'mfcc':
            freeze_list = ['final_proj', 'label_embs_concat',
                           'mask_emb', 'project_q', 'quantizer']
            for name, param in self.feature_extract.named_parameters():
                for freeze_val in freeze_list:
                    if freeze_val in name:
                        param.requires_grad = False
                        break

        self.instance_norm = nn.InstanceNorm1d(feat_dim)
        # self.channels = [channels] * 4 + [channels * 3]
        self.channels = [channels] * 4 + [1536]

        self.layer1 = Conv1dReluBn(
            feat_dim, self.channels[0], kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(self.channels[0], self.channels[1], kernel_size=3,
                                   stride=1, padding=2, dilation=2, scale=8, se_bottleneck_dim=128)
        self.layer3 = SE_Res2Block(self.channels[1], self.channels[2], kernel_size=3,
                                   stride=1, padding=3, dilation=3, scale=8, se_bottleneck_dim=128)
        self.layer4 = SE_Res2Block(self.channels[2], self.channels[3], kernel_size=3,
                                   stride=1, padding=4, dilation=4, scale=8, se_bottleneck_dim=128)

        # self.conv = nn.Conv1d(self.channels[-1], self.channels[-1], kernel_size=1)
        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, self.channels[-1], kernel_size=1)

        if time_pooling:
            self.pooling = AttentiveStatsPool(
                self.channels[-1], attention_channels=128, global_context_att=global_context_att)
            self.bn = nn.BatchNorm1d(self.channels[-1] * 2)
            self.linear = nn.Linear(self.channels[-1] * 2, emb_dim)

        if not self.update_extract:
            for param in self.feature_extract.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if not self.update_extract:
            # logger.info("Freezing Mean/Var of BatchNorm2D in AudioEncoder.")
            for m in self.feature_extract.modules():
                m.eval()

    def get_feat_num(self):
        self.feature_extract.eval()
        wav = [torch.randn(self.sr).to(next(self.feature_extract.parameters()).device)]
        with torch.no_grad():
            features = self.feature_extract(wav)
        select_feature = features[self.feature_selection]
        if isinstance(select_feature, (list, tuple)):
            return len(select_feature)
        else:
            return 1

    def get_feat(self, x_in):
        if self.update_extract:
            x = self.feature_extract([sample for sample in x])
        else:
            with torch.no_grad():
                if self.feat_type == 'fbank' or self.feat_type == 'mfcc':
                    x = self.feature_extract(x) + 1e-6  # B x feat_dim x time_len
                else:
                    x = self.feature_extract([sample for sample in x_in])

        if self.feat_type == 'fbank':
            x = x.log()

        if self.feat_type != "fbank" and self.feat_type != "mfcc":
            x = x[self.feature_selection]
            if isinstance(x, (list, tuple)):
                x = torch.stack(x, dim=0) # [L, B, T, C], T=audio_len/16000*50 (20ms hop)
            else:
                x = x.unsqueeze(0)
            norm_weights = F.softmax(self.feature_weight, dim=-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # [L,1,1,1]
            x = (norm_weights * x).sum(dim=0)
            x = torch.transpose(x, 1, 2) + 1e-6 # [B, C, T]

        x = self.instance_norm(x)
        return x

    def get_time_feature(self, x):
        x = self.get_feat(x) # [B, C(768), T]
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3) # [B, C(512), T]

        out = torch.cat([out2, out3, out4], dim=1) # [B, C(1536), T]
        out = self.conv(out) # [B, C(1536), T]
        return out

    def forward(self, x):
        time_feature = self.get_time_feature(x) # [B, C(1536), T]
        out = F.relu(time_feature)
        out = self.bn(self.pooling(out)) # [B, C(1536)]
        out = self.linear(out) # [B, C(256)]

        return out, time_feature

class ECAPA_pre_model(nn.Module):
    def __init__(self, config, num_way, num_shot):
        super().__init__()

        self.num_way = num_way
        self.num_shot = num_shot
        self.use_visual = config.use_visual
        self.test_normalize = config.test_normalize
        self.take_middle_feature = config.take_middle_feature
        self.distance = config.distance
        self.fix_audio = config.fix_audio
        self.fix_visual = config.fix_visual

        audio_emb_dim = 256
        self.audio_feature_model = ECAPA_TDNN(feat_dim=768, channels=512, emb_dim=audio_emb_dim, feat_type='wavlm_base_plus',
                                        sr=16000, feature_selection="hidden_states", update_extract=False, config_path=None,)
                                        # time_pooling=config.audio_time_pooling)
        self.load_ECAPA_feature_model(config, time_pooling=True)
        if self.fix_audio:
            for param in self.audio_feature_model.parameters():
                    param.requires_grad = False

        self.visual_exist = True
        self.visualFrontend  = visualFrontend(config.visual) # Visual Frontend, 11.216 M param
        self.visualTCN       = visualTCN()      # Visual Temporal Network TCN, 1.328 M param
        self.visualConv1D    = visualConv1D()   # Visual Temporal Network Conv1d, 0.6890 M param
        self.load_visual_feature_model(config, time_pooling=config.talknet_time_pooling)
        if self.fix_visual:
            for param in self.visualFrontend.parameters():
                    param.requires_grad = False
            for param in self.visualTCN.parameters():
                    param.requires_grad = False
            for param in self.visualConv1D.parameters():
                    param.requires_grad = False

        feature_dim = 128
        step_size = int(config.audio_num_frame*2/config.visual_sample_num_frame)
        assert step_size==config.audio_num_frame*2/config.visual_sample_num_frame
        self.audio_downsample_conv = nn.Conv1d(
            self.audio_feature_model.channels[-1], feature_dim, step_size, stride=step_size, padding=0)

        # Audio-visual Cross Attention
        self.crossA2V = attentionLayer(d_model = feature_dim, nhead = 8, positional_emb_flag=config.positional_emb_flag)
        self.crossV2A = attentionLayer(d_model = feature_dim, nhead = 8, positional_emb_flag=config.positional_emb_flag)

        # Audio-visual Self Attention
        self.selfAV = attentionLayer(d_model = feature_dim*2, nhead = 8, positional_emb_flag=config.positional_emb_flag)
        
        # visible token
        self.visible_token = nn.Embedding(2, feature_dim) # [2, 128]

        loss_cls = LOSS_DICT[config.loss_param.type]
        self.criterion = loss_cls(config.loss_param.param)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.fix_audio:
            for m in self.audio_feature_model.modules():
                m.eval()
        if self.visual_exist: 
            if self.fix_visual:
                for m in self.visualFrontend.modules():
                    m.eval()
                for m in self.visualTCN.modules():
                    m.eval()
                for m in self.visualConv1D.modules():
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

    def load_visual_feature_model(self, config, time_pooling):
        if config.visual_load_checkpoint:
            state_dict = torch.load(config.visual_checkpoint, map_location=lambda storage, loc: storage)
            visualFrontend_state_dict = OrderedDict()
            visualTCN_state_dict = OrderedDict()
            visualConv1D_state_dict = OrderedDict()
            for k,v in state_dict.items():
                if k.startswith('model.visualFrontend'):
                    visualFrontend_state_dict[k.replace('model.visualFrontend.', '')] = v
                elif k.startswith('model.visualTCN'):
                    visualTCN_state_dict[k.replace('model.visualTCN.', '')] = v
                elif k.startswith('model.visualConv1D'):
                    visualConv1D_state_dict[k.replace('model.visualConv1D.', '')] = v
                else:
                    pass
            if time_pooling:
                raise NotImplementedError('visual pooling not implemented')
            self.visualFrontend.load_state_dict(visualFrontend_state_dict, strict=True) 
            self.visualTCN.load_state_dict(visualTCN_state_dict, strict=True)
            self.visualConv1D.load_state_dict(visualConv1D_state_dict, strict=True)
            log.info('Load checkpoint for visual_pre feature model from: %s'%config.visual_checkpoint)
        else:
            log.info('Did not load checkpoint for visual_pre feature model')    

    def forward_visual_frontend(self, frames):
        B, T, C, W, H = frames.shape
        x = self.visualFrontend(frames) # [B, T, 512]
        x = x.transpose(1,2) # [B, 512, T]
        x = self.visualTCN(x) # [B, 512, T]
        x = self.visualConv1D(x) # [B, 128, T]
        x = x.transpose(1,2) # [B, T, 128]
        return x

    def forward_cross_attention(self, x1, x2):
        # [B*40, 50, 128], (B*num_way*num_shot, num_frame, 128)
        x1_c = self.crossA2V(src = x1, tar = x2)
        x2_c = self.crossV2A(src = x2, tar = x1)
        return x1_c, x2_c

    def forward_audio_visual_backend(self, x1, x2):
        x = torch.cat((x1,x2), 2)
        x = self.selfAV(src = x, tar = x)
        x = torch.reshape(x, (x1.shape[0], x1.shape[1], 256))
        return x

    def forward(self, batch, mode):
        output = Sample()

        if mode=='train' or mode=='extract':
            audio = batch.audio # [B, 40, num_segment, 32000] (B, num_way*num_shot, num_segment, 32000)
            B, num_way_shot, num_segment, L = audio.shape
            audio = audio.reshape([B*num_way_shot*num_segment, L])
            if not self.use_visual:
                audio_feature = self.audio_feature_model(audio) # [B*num_way*num_shot*num_segment, C]
                AV_feature = audio_feature.reshape([B, num_way_shot, num_segment, audio_feature.shape[-1]]) # [B, num_way*num_shot (40), num_segment, C]
            else:
                audio_feature_pre, audio_feature_time = self.audio_feature_model(audio)
                 # [B*num_way*num_shot*num_segment, C, T] (B*20*1, 1536, 99)
                audio_feature_pre = audio_feature_pre.detach()
                audio_feature_time = audio_feature_time.detach()

                audio_feature_time = F.pad(audio_feature_time, (0,1), 'replicate') # (B*20*1, 1536, 100)
                audio_feature_time = self.audio_downsample_conv(audio_feature_time) # (B*20*1, 128, 20)
                audio_feature_time = audio_feature_time.transpose(1, 2) # [B*num_way*num_shot*num_segment, T, C] (B*20*1, 20, 128)

                _, _, _, num_frame, C, W, H = batch.frames.shape
                visual_feature_time = self.forward_visual_frontend(batch.frames.reshape(B*num_way_shot*num_segment, num_frame, C, W, H))
                        # [B*num_way*num_shot*num_segment, T, C] (B*20*1, 20, 128)
                visible = batch.visible.to(visual_feature_time.device) # [B, 20] (B, num_way*num_shot)
                token = self.visible_token(visible) # [B, 20, 128] (B, num_way*num_shot, D)
                token = token.unsqueeze(2).unsqueeze(3).expand(-1, -1, num_segment, num_frame, -1) # [B, 20, 1, 20, 128] (B, num_way*num_shot, num_segment, num_frame, D)
                token = token.reshape(B*num_way_shot*num_segment, num_frame, visual_feature_time.shape[-1]) # [B*20*num_segment, 20, 128] (B*num_way*num_shot*num_segment, T, 128)
                visual_feature_time = visual_feature_time*token # [B*20*num_segment, 20, 128] (B*num_way*num_shot*num_segment, T, 128)

                audio_feature_time, visual_feature_time = self.forward_cross_attention(audio_feature_time, visual_feature_time) # [B*num_way*num_shot*num_segment, num_frame, channel(128)]
                AV_feature = self.forward_audio_visual_backend(audio_feature_time, visual_feature_time)  # [B*num_way*num_shot*num_segment, num_frame, channel(128)*2]
                AV_feature = torch.mean(AV_feature, dim=1) # [B*num_way*num_shot*num_segment, channel(128)*2]

                AV_feature = AV_feature.reshape([B, num_way_shot, num_segment, AV_feature.shape[-1]]) # [B, num_way*num_shot(20), num_segment, C]

            if mode=='train':
                assert num_segment==1
                AV_feature = AV_feature.squeeze(2)
                AV_feature = AV_feature.reshape(B, self.num_way, self.num_shot, AV_feature.shape[-1]) # [B, num_way, num_shot, C]
                if self.loss_type=='feature':
                    loss, prec1 =  self.cirterion(AV_feature, audio_feature_pre)
                else:
                    loss, prec1 =  self.cirterion(AV_feature, batch.targets)
                
                output.loss = loss
                output.prec1 = prec1

            elif mode=='extract':
                # assert num_way_shot==1
                # AV_feature = AV_feature.squeeze(1) # [B, num_segment, channel(128)*2]
                output.av_feature = AV_feature # [B, num_way*num_shot, num_segment, channel(128)*2]
                if self.loss_type=='feature':
                    loss, prec1 = self.cirterion(AV_feature, audio_feature_pre)
                    output.prec1 = prec1

        elif mode=='distance':
            av_feature_i = batch['av_feature_i'] # [B, num_segment, C]
            av_feature_j = batch['av_feature_j']

            if self.test_normalize:
                av_feature_i = F.normalize(av_feature_i, p=2, dim=2)
                av_feature_j = F.normalize(av_feature_j, p=2, dim=2)

            num_segment = av_feature_i.shape[1]

            if self.take_middle_feature:
                av_feature_i = av_feature_i[:, num_segment//2:(num_segment//2+1), :] # [B, 1, C]
                av_feature_j = av_feature_j[:, num_segment//2:(num_segment//2+1), :]
                num_segment = 1
                
            av_feature_i_re = av_feature_i.unsqueeze(2).repeat(1, 1, num_segment, 1) # [B, num_segment, num_segment, C]
            av_feature_j_re = av_feature_j.unsqueeze(1).repeat(1, num_segment, 1, 1) # [B, num_segment, num_segment, C]
            if self.distance=='cosine':
                distance = F.cosine_similarity(av_feature_i_re, av_feature_j_re, dim=3) # [B, num_segment, num_segment]
                distance = (distance+1)/2.0 # 0~1
            elif self.distance=='L2':
                distance = F.pairwise_distance(av_feature_i_re, av_feature_j_re) # [B, num_segment, num_segment]
                distance = (2-distance)/2.0
            distance = distance.mean(2).mean(1) # [B]

            output.scores = distance
        else:
            raise NotImplementedError()
        
        return output

