from model import common

import torch.nn as nn
import torch

def make_model(args, parent=False):
    return EMSR(args)



class ResStage(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=0.1):

        super(ResStage, self).__init__()
        
        m = nn.ModuleList()
        for _ in range(4):
            m.append(common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale)
            )
                     
        self.body = nn.Sequential(*m)
        

    def forward(self, x):
        x = self.body(x)

        return x

    

class EMSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EMSR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        
        if args.act == 'mish':
            act = common.Mish()
        elif args.act == 'swish':
            act = common.Swish()
        elif args.act == 'prelu':
            act = nn.PReLU(True)
        elif args.act == 'leakyrelu':
            act = nn.LeakyReLU(0.01, True)
        else:
            act = nn.ReLU(True)
        print(act)
        
        
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        
        m_body = nn.ModuleList()
        for _ in range(8):
            m_body.append(
                ResStage(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale)
            )
            
#         m_body_stage1= [
#             common.ResBlock(
#                 conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
#             ) for _ in range(4) 
#         ]
#         m_body_stage2= [
#             common.ResBlock(
#                 conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
#             ) for _ in range(4) 
#         ]
#         m_body_stage3= [
#             common.ResBlock(
#                 conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
#             ) for _ in range(4) 
#         ]
#         m_body_stage4= [
#             common.ResBlock(
#                 conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
#             ) for _ in range(4) 
#         ]
#         m_body_stage5= [
#             common.ResBlock(
#                 conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
#             ) for _ in range(4) 
#         ]
#         m_body_stage6= [
#             common.ResBlock(
#                 conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
#             ) for _ in range(4) 
#         ]
#         m_body_stage7= [
#             common.ResBlock(
#                 conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
#             ) for _ in range(4) 
#         ]
#         m_body_stage8= [
#             common.ResBlock(
#                 conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
#             ) for _ in range(4) 
#         ]
        
        m_body_reduction = conv(n_feats*8, n_feats, 1)
       
        
        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.body_reduction = m_body_reduction
        self.tail = nn.Sequential(*m_tail)
        
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = x
                     
        out_feats = []
        for i in range(8):
            x = self.body[i](x)
            out_feats.append(x)
        fusion = torch.cat(out_feats,1)
                     
        x = self.body_reduction(fusion)
        x += res
        x = self.tail(x)
        x = self.add_mean(x)

        return x 


