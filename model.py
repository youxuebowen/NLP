#!/usr/bin/python
# -*-coding:utf-8-*-
# @Time:2021/10/30  21:13
# @Author:Mr.Wan
# @Email:2215225145@qq.com
# @File:model.py
# Software:PyCharm
import torch
from torch import nn
import torch.nn.functional as F
import config


class TextCNN(nn.Module):
    # output_size为输出类别（2个类别，0和1）,三种kernel，size分别是3,4，5，每种kernel有100个
    def __init__(self, vocab_size, embedding_dim, output_size, vectors, filter_num=100, kernel_list=(3, 4, 5), dropout=0.5):
        super(TextCNN, self).__init__()
        chanel_num = 1
        # embedding层，转化为固定长度
        # self.args = config.args
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if config.args.static:
            self.embedding = self.embedding.from_pretrained(vectors, freeze=not config.args.non_static)
        if config.args.multichannel:
            self.embedding2 = nn.Embedding(vocab_size, embedding_dim).from_pretrained(vectors)
            chanel_num += 1
        else:
            self.embedding2 = None
        # 1表示channel_num，filter_num即输出数据通道数，卷积核大小为(kernel, embedding_dim)，进行二维卷积
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(chanel_num, filter_num, (kernel, embedding_dim)),
                          nn.LeakyReLU(),
                          nn.MaxPool2d((config.MAX_SENTENCE_SIZE - kernel + 1, 1)))
            for kernel in kernel_list
        ])
        # 通过一个线性层，获得最终输出
        self.fc = nn.Linear(filter_num * len(kernel_list), output_size)
        # Dropout防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.embedding2:
            x = torch.stack([self.embedding(x), self.embedding2(x)], dim=1)
        else:
            x = self.embedding(x)
            x = x.unsqueeze(1)
        # x = self.embedding(x)  # [128, 50, 200] (batch, seq_len, embedding_dim)
        # x = x.unsqueeze(1)  # [128, 1, 50, 200] 即(batch, channel_num, seq_len, embedding_dim)
        # out = [conv(x) for conv in self.convs]
        # out = torch.cat(out, dim=1)  # [128, 300, 1, 1]，各通道的数据拼接在一起
        # out = out.view(x.size(0), -1)  # 展平
        # out = self.dropout(out)  # 构建dropout层
        # logits = self.fc(out)  # 结果输出[128, 2]
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
