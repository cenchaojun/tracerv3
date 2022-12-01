"""
author: Min Seok Lee and Wooseok Shin
Github repo: https://github.com/Karel911/TRACER
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.EfficientNet import EfficientNet
from util.effi_utils import get_model_shape
from modules.att_modules import RFB_Block, aggregation, ObjectAttention
import argparse
from res import resnet18

def getConfig():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='train', help='Model Training or Testing options')
    parser.add_argument('--exp_num', default=0, type=str, help='experiment_number')
    parser.add_argument('--dataset', type=str, default='DUTS', help='DUTS')
    parser.add_argument('--data_path', type=str, default='data/')

    # Model parameter settings
    parser.add_argument('--arch', type=str, default='0', help='Backbone Architecture')
    #parser.add_argument('--arch', type=str, default='7', help='Backbone Architecture')
    parser.add_argument('--channels', type=list, default=[64, 128, 256, 512])
    parser.add_argument('--RFB_aggregated_channel', type=int, nargs='*', default=[32, 64, 128])
    parser.add_argument('--frequency_radius', type=int, default=16, help='Frequency radius r in FFT')
    parser.add_argument('--denoise', type=float, default=0.93, help='Denoising background ratio')
    parser.add_argument('--gamma', type=float, default=0.1, help='Confidence ratio')

    # Training parameter settings
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--criterion', type=str, default='API', help='API or bce')
    parser.add_argument('--scheduler', type=str, default='Reduce', help='Reduce or Step')
    parser.add_argument('--aug_ver', type=int, default=2, help='1=Normal, 2=Hard')
    parser.add_argument('--lr_factor', type=float, default=0.1)
    parser.add_argument('--clipping', type=float, default=2, help='Gradient clipping')
    parser.add_argument('--patience', type=int, default=5, help="Scheduler ReduceLROnPlateau's parameter & Early Stopping(+5)")
    parser.add_argument('--model_path', type=str, default='results/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_map', type=bool, default=None, help='Save prediction map')


    # Hardware settings
    parser.add_argument('--multi_gpu', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    cfg = parser.parse_args()

    return cfg

class TRACER(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        #self.model = EfficientNet.from_pretrained(f'efficientnet-b{cfg.arch}', advprop=True)
        #self.block_idx, self.channels = get_model_shape()  # self.channel[24, 40, 112, 320] channel[32, 64, 128]
        self.backbone = resnet18()
        self.backbone.load_state_dict(torch.load('./pretrain/resnet18-5c106cde.pth'), strict=False)
        self.channels = [64, 128, 256, 512]

        # Receptive Field Blocks

        channels = [int(arg_c) for arg_c in cfg.RFB_aggregated_channel]
        self.rfb2 = RFB_Block(self.channels[1], channels[0])
        self.rfb3 = RFB_Block(self.channels[2], channels[1])
        self.rfb4 = RFB_Block(self.channels[3], channels[2])

        # Multi-level aggregation
        self.agg = aggregation(channels)

        # Object Attention
        self.ObjectAttention2 = ObjectAttention(channel=self.channels[1], kernel_size=3)
        self.ObjectAttention1 = ObjectAttention(channel=self.channels[0], kernel_size=3)

    def forward(self, inputs):
        B, C, H, W = inputs.size()

        # EfficientNet backbone Encoder
        # x = self.model.initial_conv(inputs)
        features = self.backbone(x)

        x3_rfb = self.rfb2(features[1])
        x4_rfb = self.rfb3(features[2])
        x5_rfb = self.rfb4(features[3])

        D_0 = self.agg(x5_rfb, x4_rfb, x3_rfb)

        ds_map0 = F.interpolate(D_0, scale_factor=8, mode='bilinear')

        D_1 = self.ObjectAttention2(D_0, features[1])
        ds_map1 = F.interpolate(D_1, scale_factor=8, mode='bilinear')

        ds_map = F.interpolate(D_1, scale_factor=2, mode='bilinear')
        D_2 = self.ObjectAttention1(ds_map, features[0])
        ds_map2 = F.interpolate(D_2, scale_factor=4, mode='bilinear')

        final_map = (ds_map2 + ds_map1 + ds_map0) / 3

        # return torch.sigmoid(final_map), torch.sigmoid(edge), \
        #        (torch.sigmoid(ds_map0), torch.sigmoid(ds_map1), torch.sigmoid(ds_map2))

        return torch.sigmoid(final_map),  \
               (torch.sigmoid(ds_map0), torch.sigmoid(ds_map1), torch.sigmoid(ds_map2))

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    args = getConfig()
    models = TRACER(args).cuda()
    x = torch.randn(2, 3, 320, 320).cuda()
    print(models(x))