# -*- ecoding: utf-8 -*-
# @ModuleName: conv2d_test
# @Function: 
# @Author: Yufan-tyf
# @Time: 2022/3/4 7:22 PM
import torch
import torch.nn as nn

if __name__ == '__main__':
    sample = torch.zeros([37, 1, 64, 64])
    sampleConv = nn.Conv2d(1, 200, (2, 64))(sample)
    embed = nn.Embedding(315, 64, padding_idx=0)
