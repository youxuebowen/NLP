#!/usr/bin/python
# -*-coding:utf-8-*-
# @Time:2021/11/11  21:27
# @Author:Mr.Wan
# @Email:2215225145@qq.com
# @File:model2.py
# Software:PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    # def __init__(self, args):
    #     super(TextCNN, self).__init__()
    def __init__(self, vocab_size, embedding_dim, output_size, filter_num=100, kernel_list=(3, 4, 5), dropout=0.5):
        super(TextCNN, self).__init__()
        # self.args = args

        # class_num = args.class_num
        # chanel_num = 1
        # filter_num = args.filter_num
        # filter_sizes = args.filter_sizes
        #
        # vocabulary_size = args.vocabulary_size
        # embedding_dimension = args.embedding_dim
        # self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # if args.static:
        #     self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.non_static)
        # if args.multichannel:
        #     self.embedding2 = nn.Embedding(vocabulary_size, embedding_dimension).from_pretrained(args.vectors)
        #     chanel_num += 1
        # else:
        #     self.embedding2 = None
        # self.convs = nn.ModuleList(
        #     [nn.Conv2d(chanel_num, filter_num, (size, embedding_dimension)) for size in filter_sizes])
        self.embedding2 = None
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 100, (size, embedding_dim)) for size in kernel_list])
        # self.dropout = nn.Dropout(args.dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_list) * filter_num, 1)

    def forward(self, x):
        if self.embedding2:
            x = torch.stack([self.embedding(x), self.embedding2(x)], dim=1)
        else:
            x = self.embedding(x)
            x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
