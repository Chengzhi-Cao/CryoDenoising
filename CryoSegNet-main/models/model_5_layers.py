# UNET model with 5 layers
 
import torch
import torch.nn as nn

import numpy as np
import os

import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        s = self.conv(x)
        p = self.pool(s)
        return s, p

class AttentionGate(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv2d(in_c[0], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(in_c[1], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, g, s):
        Wg = self.Wg(g)
        Ws = self.Ws(s)
        out = self.relu(Wg + Ws)
        out = self.output(out)
        return out * s

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.ag = AttentionGate(in_c, out_c)
        self.c1 = ConvBlock(in_c[0]+out_c, out_c)

    def forward(self, x, s):
        x = self.up(x)
        s = self.ag(x, s)
        x = torch.cat([x, s], axis=1)
        x = self.c1(x)
        return x

class UNET(nn.Module):
    def __init__(self):
        super().__init__()

        self.e1 = EncoderBlock(1, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)
        self.e5 = EncoderBlock(512, 1024)
        
        self.b1 = ConvBlock(1024, 2048)

        self.d1 = DecoderBlock([2048, 1024], 1024)
        self.d2 = DecoderBlock([1024, 512], 512)
        self.d3 = DecoderBlock([512, 256], 256)
        self.d4 = DecoderBlock([256, 128], 128)
        self.d5 = DecoderBlock([128, 64], 64)

        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, x, file_path):       # x=[1,1,1024,1024]
        s1, p1 = self.e1(x)     # p1=[1,64,512,512], s1=[1,64,1024,1024]

        for i in range(len(p1[0,:,0,0])):
            if i % 2 == 0:
                _fea = p1[0,i,:,:].cpu().data.numpy()
                _a = np.clip(_fea, 0, 1) # 将numpy数组约束在[0, 1]范围内
                trans_prob_mat = (_a.T/np.sum(_a, 1)).T
                df = pd.DataFrame(trans_prob_mat)
                plt.figure()
                ax = sns.heatmap(df, cmap='jet', cbar=False)
                plt.xticks(alpha=0)
                plt.tick_params(axis='x', width=0)
                plt.yticks(alpha=0)
                plt.tick_params(axis='y', width=0)
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)

                save_visual_path = os.path.join(file_path,'p1') 
                # _save_path = os.path.join(save_visual_path)
                if not os.path.exists(save_visual_path):
                    os.makedirs(save_visual_path)
                output_path = os.path.join(save_visual_path,'feat{}.jpg'.format(i))
                plt.savefig(output_path, transparent=True)   


        s2, p2 = self.e2(p1)    # p2=[1,128,256,256], s1=[1,128,512,512]

        for i in range(len(p2[0,:,0,0])):
            if i % 2 == 0:
                _fea = p2[0,i,:,:].cpu().data.numpy()
                _a = np.clip(_fea, 0, 1) # 将numpy数组约束在[0, 1]范围内
                trans_prob_mat = (_a.T/np.sum(_a, 1)).T
                df = pd.DataFrame(trans_prob_mat)
                plt.figure()
                ax = sns.heatmap(df, cmap='jet', cbar=False)
                plt.xticks(alpha=0)
                plt.tick_params(axis='x', width=0)
                plt.yticks(alpha=0)
                plt.tick_params(axis='y', width=0)
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)

                save_visual_path = os.path.join(file_path,'p2') 
                # _save_path = os.path.join(save_visual_path)
                if not os.path.exists(save_visual_path):
                    os.makedirs(save_visual_path)
                output_path = os.path.join(save_visual_path,'feat{}.jpg'.format(i))
                plt.savefig(output_path, transparent=True) 
                
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        s5, p5 = self.e5(p4)

        b1 = self.b1(p5)

        d1 = self.d1(b1, s5)
        d2 = self.d2(d1, s4)
        d3 = self.d3(d2, s3)
        
        for i in range(len(d3[0,:,0,0])):

            if i % 2 == 0:
                _fea = d3[0,i,:,:].cpu().data.numpy()
                _a = np.clip(_fea, 0, 1) # 将numpy数组约束在[0, 1]范围内
                trans_prob_mat = (_a.T/np.sum(_a, 1)).T
                df = pd.DataFrame(trans_prob_mat)
                plt.figure()
                ax = sns.heatmap(df, cmap='jet', cbar=False)
                plt.xticks(alpha=0)
                plt.tick_params(axis='x', width=0)
                plt.yticks(alpha=0)
                plt.tick_params(axis='y', width=0)
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)

                save_visual_path = os.path.join(file_path,'d3') 
                # _save_path = os.path.join(save_visual_path)
                if not os.path.exists(save_visual_path):
                    os.makedirs(save_visual_path)
                output_path = os.path.join(save_visual_path,'feat{}.jpg'.format(i))
                plt.savefig(output_path, transparent=True)   


        
        d4 = self.d4(d3, s2)    # d5=[1,128,512,512]
        
        for i in range(len(d4[0,:,0,0])):

            if i % 2 == 0:
                _fea = d4[0,i,:,:].cpu().data.numpy()
                _a = np.clip(_fea, 0, 1) # 将numpy数组约束在[0, 1]范围内
                trans_prob_mat = (_a.T/np.sum(_a, 1)).T
                df = pd.DataFrame(trans_prob_mat)
                plt.figure()
                ax = sns.heatmap(df, cmap='jet', cbar=False)
                plt.xticks(alpha=0)
                plt.tick_params(axis='x', width=0)
                plt.yticks(alpha=0)
                plt.tick_params(axis='y', width=0)
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)

                save_visual_path = os.path.join(file_path,'d4') 
                # _save_path = os.path.join(save_visual_path)
                if not os.path.exists(save_visual_path):
                    os.makedirs(save_visual_path)
                output_path = os.path.join(save_visual_path,'feat{}.jpg'.format(i))
                plt.savefig(output_path, transparent=True)   


        
        d5 = self.d5(d4, s1)    # d5=[1,64,1024,1024]



        for i in range(len(d5[0,:,0,0])):

            if i % 2 == 0:
                _fea = d5[0,i,:,:].cpu().data.numpy()
                _a = np.clip(_fea, 0, 1) # 将numpy数组约束在[0, 1]范围内
                trans_prob_mat = (_a.T/np.sum(_a, 1)).T
                df = pd.DataFrame(trans_prob_mat)
                plt.figure()
                ax = sns.heatmap(df, cmap='jet', cbar=False)
                plt.xticks(alpha=0)
                plt.tick_params(axis='x', width=0)
                plt.yticks(alpha=0)
                plt.tick_params(axis='y', width=0)
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)

                save_visual_path = os.path.join(file_path,'d5') 
                # _save_path = os.path.join(save_visual_path)
                if not os.path.exists(save_visual_path):
                    os.makedirs(save_visual_path)
                output_path = os.path.join(save_visual_path,'feat{}.jpg'.format(i))
                plt.savefig(output_path, transparent=True)   




        output = self.output(d5)
        return output