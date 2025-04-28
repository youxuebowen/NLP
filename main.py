#!/usr/bin/python
# -*-coding:utf-8-*-
# @Time:2021/10/30  20:55
# @Author:Mr.Wan
# @Email:2215225145@qq.com
# @File:dataloader.py
# Software:PyCharm
import re  # 正则表达
from torchtext.legacy import data
import jieba
import logging
from torchtext.vocab import Vectors
import config
import torch
import random
from torch import nn, optim
import model
import utils
import time
import matplotlib.pyplot as plt

jieba.setLogLevel(logging.INFO)  # 为了不显示错误信息
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')


def word_cut(text):
    text = regex.sub(' ', text)
    return [word for word in jieba.cut(text) if word.strip()]


def get_dataset(corpus_path, text_field, label_field, datatype):
    """
    构建torchtext数据集
    :param corpus_path: 数据路径
    :param text_field: torchtext设置的文本域
    :param label_field: torchtext设置的文本标签域
    :param datatype: 文本的类别
    :return: torchtext格式的数据集以及设置的域
    """
    fields = [('text', text_field), ('label', label_field)]
    examples = []
    with open(corpus_path, encoding='utf-8') as reader:
        for line in reader:
            content = line.rstrip()
            # if datatype == 'pos':
            if datatype == '1.1房地产买卖合同.txt':
                label = 0
            if datatype == '1.2一般商品买卖合同.txt':
                label = 1
            if datatype == '1.3一般买卖合同（合同+订单）.txt':
                label = 3
            if datatype == '1.4代理经销、分销合同.txt':
                label = 4
            if datatype == '1.5超市与供应商购销合同.txt':
                label = 5
            if datatype == '1.6采购、买卖配套文本.txt':
                label = 6
            if datatype == '1.7特殊模式买卖合同.txt':
                label = 7
            # examples.append(data.Example.fromlist([content[:-2], label], fields))
            examples.append(data.Example.fromlist([content, label], fields))
    return examples, fields


def Dataloader():
    # 文本内容，tokenize设置一个tokenize分词器给Field用,使用自定义的分词方法，将内容转换为小写，设置最大长度等
    # TEXT = data.Field(tokenize=jieba.lcut, lower=True, fix_length=config.MAX_SENTENCE_SIZE, batch_first=True)
    # 通过Field把文本数据转化为tensor类型，可处理
    TEXT = data.Field(sequential=True, tokenize=word_cut, fix_length=config.MAX_SENTENCE_SIZE,
                      batch_first=True)
    # 文本对应的标签
    LABEL = data.LabelField(dtype=torch.float)

    # 构建data数据，让标签数值化，label=1为pos
    # 对于文件路径可以有优化，可以直接使用相对地址
    # pos_examples, pos_fields = dataloader.get_dataset(config.POS_CORPUS_PATH, TEXT, LABEL, 'pos')
    examples_1, labels_1 = get_dataset(config.PATH_1, TEXT, LABEL, '1.1房地产买卖合同.txt')
    # neg_examples, neg_fields = dataloader.get_dataset(config.NEG_CORPUS_PATH, TEXT, LABEL, 'neg')
    examples_2, labels_2 = get_dataset(config.PATH_2, TEXT, LABEL, '1.2一般商品买卖合同.txt')
    examples_3, labels_3 = get_dataset(config.PATH_3, TEXT, LABEL, '1.3一般买卖合同（合同+订单）.txt')
    examples_4, labels_4 = get_dataset(config.PATH_4, TEXT, LABEL, '1.4代理经销、分销合同.txt')
    examples_5, labels_5 = get_dataset(config.PATH_5, TEXT, LABEL, '1.5超市与供应商购销合同.txt')
    examples_6, labels_6 = get_dataset(config.PATH_6, TEXT, LABEL, '1.6采购、买卖配套文本.txt')
    examples_7, labels_7 = get_dataset(config.PATH_7, TEXT, LABEL, '1.7特殊模式买卖合同.txt')
    # all_examples, all_fields = examples_1 + examples_2 + examples_3 + examples_4 + examples_5 + examples_6 +
    # examples_7, \ labels_1 + labels_2 + labels_3 + labels_4 + labels_5 + labels_6 + labels_7
    all_examples, all_fields = examples_1 + examples_2 + examples_3 + examples_4, labels_1 + labels_2 + labels_3 + labels_4
    # 构建torchtext类型的数据集
    total_data = data.Dataset(all_examples, all_fields)
    # 数据集切分
    train_data, test_data = total_data.split(random_state=random.seed(config.RANDOM_SEED), split_ratio=0.8)
    # 切分后的数据查看
    # 数据维度查看
    print('len of train data: %r' % len(total_data))  # len of train data: 8530
    print('len of train data: %r' % len(train_data))  # len of train data: 8530
    print('len of test data: %r' % len(test_data))  # len of test data: 2132

    # def load_word_vectors(model_name, model_path):
    #     vectors = Vectors(name=model_name, cache=model_path)
    #     return vectors

    def load_dataset(train_data, test_data, TEXT, LABEL, args1, **kwargs):
        if args1.static and config.pretrained_name and args1.pretrained_path:
            # vectors = load_word_vectors(args1.pretrained_name, args1.pretrained_path)
            TEXT.build_vocab(train_data, test_data, vectors='glove.6B.300d')
        else:
            # TEXT.build_vocab(train_data, test_data)
            TEXT.build_vocab(train_data)
        LABEL.build_vocab(train_data)
        train_iterator, test_iterator = data.BucketIterator.splits((train_data, test_data),
                                                                   batch_size=config.BATCH_SIZE,
                                                                   sort=False)
        return train_iterator, test_iterator

    if config.args.multichannel:
        config.args.static = True
        config.args.non_static = True
    train_iter, dev_iter = load_dataset(train_data, test_data, TEXT, LABEL, config.args, device=-1, repeat=False, shuffle=True)
    # 使用预训练词向量
    if config.args.static:
        config.args.EMBEDDING_SIZE = TEXT.vocab.vectors.size()[-1]
        vectors = TEXT.vocab.vectors
    # 创建模型
    # text_cnn = model.TextCNN(len(TEXT.vocab), config.args.EMBEDDING_SIZE, len(LABEL.vocab), vectors).to(device)
    text_cnn = model.TextCNN(len(TEXT.vocab), config.args.EMBEDDING_SIZE, len(LABEL.vocab), vectors=vectors).to(device)
    if config.args.snapshot:
        print('\nLoading model from {}...\n'.format(config.args.snapshot))
        text_cnn.load_state_dict(torch.load(config.args.snapshot))
    # 选取优化器，设置学习率
    optimizer = optim.Adam(text_cnn.parameters(), lr=config.LEARNING_RATE)
    # 选取损失函数
    criterion = nn.CrossEntropyLoss()
    # # 抽一条数据查看
    # print(train_data.examples[10].text)
    # ['never', 'engaging', ',', 'utterly', 'predictable', 'and', 'completely', 'void', 'of', 'anything', 'remotely',
    # 'interesting', 'or', 'suspenseful']
    # print(train_data.examples[10].label)
    # 0
    # 为该样本数据构建字典，并将每个单词映射到对应数字
    # TEXT.build_vocab(train_data)
    # LABEL.build_vocab(train_data)
    # 构建迭代(iterator)类型的数据
    # train_iterator, test_iterator = data.BucketIterator.splits((train_data, test_data),
    #                                                            batch_size=config.BATCH_SIZE,
    #                                                            sort=False)
    # 查看字典长度
    # print(len(LABEL.vocab))  # 19206
    # 查看字典中前10个词语
    # print(TEXT.vocab.itos[:145])  # ['<unk>', '<pad>', ',', 'the', 'a', 'and', 'of', 'to', '.', 'is']
    # 查找'name'这个词对应的词典序号, 本质是一个dict
    # print(TEXT.vocab.stoi['合同'])
    # 绘制结果
    model_train_acc, model_test_acc = [], []
    start = time.time()
    # 模型训练
    for epoch in range(config.EPOCH):
        # 不断训练
        train_acc = utils.train(text_cnn, train_iter, optimizer, criterion, dev_iter, epoch)
        print("epoch = {}, 训练准确率={}".format(epoch + 1, train_acc))

        test_acc = utils.evaluate(text_cnn, dev_iter)
        print("epoch = {}, 测试准确率={}".format(epoch + 1, test_acc))

        model_train_acc.append(train_acc)
        model_test_acc.append(test_acc)

    print('total train time:', time.time() - start)
    # 绘制训练过程
    plt.plot(model_train_acc)
    plt.plot(model_test_acc)
    plt.ylim(ymin=0.5, ymax=1.01)
    plt.title("The accuracy of textCNN model")
    plt.legend(['train', 'test'])
    plt.show()


if __name__ == '__main__':
    Dataloader()
