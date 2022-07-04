import collections
import json
import os
import random
import subprocess
import sys
from typing import Any, Dict, Union
from matplotlib import pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import torch
import glog as log
from torch.utils.data import DataLoader
import torch.nn as nn
from collections import OrderedDict, defaultdict
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from intervaltree import IntervalTree

def download_from_google_drive(gd_id, destination):
    """
    Use the requests package to download a file from Google Drive.
    """
    save_dir = os.path.dirname(destination)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    cmd = f'gdown -O {destination} --id {gd_id}'
    subprocess.call(cmd, shell=True)

def load_pretrained_model(checkpoint_path, *args, **kwargs):
    with open(checkpoint_path, "rb") as f:
        ckpt = torch.load(f, map_location=lambda storage, loc: storage)
    return ckpt

def load_ckpt(ckpt_pth, model, optimizer, lr_schedular, score_name):
    ckpt = torch.load(ckpt_pth, map_location=next(model.parameters()).device)

    state_dict = ckpt['model']
    if isinstance(model, nn.DataParallel) or isinstance(model, DistributedDataParallel):
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = 'module.' + k
        #     new_state_dict[name] = v
        model = model.module
    # else:
    #     new_state_dict = state_dict

    model.load_state_dict(state_dict, strict=True)
    if optimizer:
        optimizer.load_state_dict(ckpt['optimizer'])
    if lr_schedular:
        lr_schedular.load_state_dict(ckpt['lr_schedular'])

    if score_name:
        score = ckpt[score_name]
    else:
        score_name = 'none'
        score = -1

    iteration = ckpt['iteration']

    log.info('Loaded checkpoint from %s, iteration %d, %s %.4f'%(ckpt_pth, iteration, score_name, score))

    return model, optimizer, lr_schedular, iteration

