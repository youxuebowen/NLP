#!/usr/bin/python
# -*-coding:utf-8-*-
# @Time:2021/10/30  20:58
# @Author:Mr.Wan
# @Email:2215225145@qq.com
# @File:train.py
# Software:PyCharm
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
# 新版本 torchtext.legacy
# 原来是 from torchtext import data
# from torchtext.legacy import data
# import random
import config
import dataloader
import model
import utils
import time
import argparse
# from torchtext.vocab import Vectors
# import dataset
# 命令行用来传递参数

def main():
    torch.manual_seed(config.RANDOM_SEED)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # parser = argparse.ArgumentParser(description='TextCNN text classifier')
    # parser.add_argument('-static', type=bool, default=True, help='whether to use static pre-trained word vectors')
    # parser.add_argument('-non-static', type=bool, default=False,
    #                     help='whether to fine-tune static pre-trained word vectors')
    # parser.add_argument('-pretrained-name', type=str, default='sgns.zhihu.word',
    #                     help='filename of pre-trained word vectors')
    # parser.add_argument('-pretrained-path', type=str, default='pretrained', help='path of pre-trained word vectors')
    # parser.add_argument('-EMBEDDING_SIZE', type=int, default=128, help='number of embedding dimension [default: 128]')
    # parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    # parser.add_argument('-early-stopping', type=int, default=1000,
    #                     help='iteration numbers to stop without performance increasing')
    # parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
    # args = parser.parse_args()
    train_iterator, test_iterator = dataloader.Dataloader()
    # 创建模型
    # text_cnn = model.TextCNN(len(TEXT.vocab), args.EMBEDDING_SIZE, len(LABEL.vocab), *args).to(device)
    # 选取优化器，设置学习率
    # optimizer = optim.Adam(text_cnn.parameters(), lr=config.LEARNING_RATE)
    # 选取损失函数
    # criterion = nn.CrossEntropyLoss()

    # # 绘制结果
    # model_train_acc, model_test_acc = [], []
    # start = time.time()
    # # 模型训练
    # for epoch in range(config.EPOCH):
    #     train_acc = utils.train(text_cnn, train_iterator, optimizer, criterion)
    #     print("epoch = {}, 训练准确率={}".format(epoch + 1, train_acc))
    #
    #     test_acc = utils.evaluate(text_cnn, test_iterator)
    #     print("epoch = {}, 测试准确率={}".format(epoch + 1, test_acc))
    #
    #     model_train_acc.append(train_acc)
    #     model_test_acc.append(test_acc)
    #
    # print('total train time:', time.time() - start)
    # # 绘制训练过程
    # plt.plot(model_train_acc)
    # plt.plot(model_test_acc)
    # plt.ylim(ymin=0.5, ymax=1.01)
    # plt.title("The accuracy of textCNN model")
    # plt.legend(['train', 'test'])
    # plt.show()


if __name__ == '__main__':
    main()
