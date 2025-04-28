#!/usr/bin/python
# -*-coding:utf-8-*-
# @Time:2021/10/30  21:12
# @Author:Mr.Wan
# @Email:2215225145@qq.com
# @File:utils.py
# Software:PyCharm
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def en_seg(sentence):
    """
    简单的英文分词方法，
    :param sentence: 需要分词的语句
    :return: 返回分词结果
    """
    return sentence.split()


def binary_acc(pred, y):
    """
    计算模型的准确率
    :param pred: 预测值
    :param y: 实际真实值
    :return: 返回准确率
    """
    correct = torch.eq(pred, y).float()
    acc = correct.sum() / len(correct)
    return acc


def train(model, train_data, optimizer, criterion, test_data, epoch):
    """
    模型训练
    :param model: 训练的模型
    :param train_data: 训练数据
    :param optimizer: 优化器
    :param criterion: 损失函数
    :return: 该论训练各批次正确率平均值
    """
    best_acc = 0
    last_step = 0
    avg_acc = []
    model.train()  # 进入训练模式
    for i, batch in enumerate(train_data):
        pred = model(batch.text.to(device)).cpu()
        loss = criterion(pred, batch.label.long())
        acc = binary_acc(torch.max(pred, dim=1)[1], batch.label)
        avg_acc.append(acc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % config.test_interval == 0:
            # dev_acc = eval(dev_iter, model, args)
            dev_acc = evaluate(model, test_data)
            if dev_acc > best_acc:
                best_acc = dev_acc
                last_step = epoch
                if config.args.save_best:
                    print('Saving best model, acc: {:.4f}%\n'.format(best_acc * 100))
                    save(model, config.Save_dir, 'best', epoch)
            else:
                if epoch - last_step >= config.args.early_stopping:
                    print('\nearly stop by {} steps, acc: {:.4f}%'.format(config.args.early_stopping, best_acc))
                    raise KeyboardInterrupt

    # 计算所有批次数据的结果
    avg_acc = np.array(avg_acc).mean()
    return avg_acc


def evaluate(model, test_data):
    """
    使用测试数据评估模型
    :param model: 模型
    :param test_data: 测试数据
    :return: 该论训练好的模型预测测试数据，查看预测情况
    """
    avg_acc = []
    model.eval()  # 进入测试模式
    with torch.no_grad():
        for i, batch in enumerate(test_data):
            pred = model(batch.text.to(device)).cpu()
            acc = binary_acc(torch.max(pred, dim=1)[1], batch.label)
            avg_acc.append(acc)
    return np.array(avg_acc).mean()


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
