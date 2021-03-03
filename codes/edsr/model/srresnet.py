# +
from model import common
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy.random as npr
import numpy as np
import torch.nn.functional as F
import random
import math


def make_model(args, parent=False):
    return SRResNet(args)

class SRResNet(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SRResNet, self).__init__()

        n_resblocks = 5
        n_feats = 64
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.PReLU()
        
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            act
        ]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, bn=True, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))
        m_body.append(nn.BatchNorm2d(n_feats))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act='prelu'),
            nn.Conv2d(n_feats, 3, kernel_size=9, padding=4)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

        
    def forward(self, x, flag=False, hr=None):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x
        
        x = self.tail[0](res)
        if flag:
            self.eval()
            x_new = x.clone().detach()
            x_new = Variable(x_new.data, requires_grad=True).cuda()
            num_batch, num_channel, H, W = x_new.shape
            HW = H*W
            
            sr = self.tail[-1](x_new)
            criterion = nn.L1Loss()
            loss = criterion(sr, hr)
            
            self.zero_grad()
            loss.backward()
            grads_val = x_new.grad.clone().detach()
            grad_channel_mean = torch.mean(grads_val.view(num_batch, num_channel, -1), dim=2)
            channel_mean = grad_channel_mean
            grad_channel_mean = grad_channel_mean.view(num_batch, num_channel, 1, 1)
            spatial_mean = torch.sum(x_new * grad_channel_mean, 1)
            spatial_mean = spatial_mean.view(num_batch, HW)
            self.zero_grad()
            
            choose_one = random.randint(0,9)
            if choose_one <= 4:
                # ---------------------------- spatial -----------------------
                spatial_drop_num = math.ceil(HW * 1 / 3.0)
                th18_mask_value = torch.sort(spatial_mean, dim=1, descending=True)[0][:, spatial_drop_num]
                th18_mask_value = th18_mask_value.view(num_batch, 1).expand(num_batch, 36864)
                mask_all_cuda = torch.where(spatial_mean > th18_mask_value, torch.zeros(spatial_mean.shape).cuda(),
                                            torch.ones(spatial_mean.shape).cuda())
                mask_all = mask_all_cuda.reshape(num_batch, H, H).view(num_batch, 1, H, H)
            else:
                # -------------------------- channel ----------------------------
                vector_thresh_percent = math.ceil(num_channel * 1 / 3.2)
                vector_thresh_value = torch.sort(channel_mean, dim=1, descending=True)[0][:, vector_thresh_percent]
                vector_thresh_value = vector_thresh_value.view(num_batch, 1).expand(num_batch, num_channel)
                vector = torch.where(channel_mean > vector_thresh_value,
                                     torch.zeros(channel_mean.shape).cuda(),
                                     torch.ones(channel_mean.shape).cuda())
                mask_all = vector.view(num_batch, num_channel, 1, 1)
            mask_all[int(num_batch/3):,:,:,:] = 1
            self.train()
            mask_all = Variable(mask_all, requires_grad=True)
            x = x * mask_all
            
            
        x = self.tail[-1](x)
        x = self.add_mean(x)

        return x 

    
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
