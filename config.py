#!/usr/bin/python
# -*-coding:utf-8-*-
# @Time:2021/10/30  21:00
# @Author:Mr.Wan
# @Email:2215225145@qq.com
# @File:config.py
# Software:PyCharm
import argparse
# 模型相关参数
# RANDOM_SEED = 1000  # 随机数种子
RANDOM_SEED = 30  # 随机数种子
# BATCH_SIZE = 128    # 批次数据大小
BATCH_SIZE = 28    # 批次数据大小
LEARNING_RATE = 1e-3   # 学习率
EMBEDDING_SIZE = 400   # 词向量维度
# MAX_SENTENCE_SIZE = 20  # 设置最大语句长度
MAX_SENTENCE_SIZE = 300  # 设置最大语句长度
EPOCH = 100           # 训练测轮次

# 命令行
parser = argparse.ArgumentParser(description='TextCNN text classifier')
parser.add_argument('-early-stopping', type=int, default=1000,
                    help='iteration numbers to stop without performance increasing')
parser.add_argument('-static', type=bool, default=True, help='whether to use static pre-trained word vectors')
parser.add_argument('-non-static', type=bool, default=True,
                        help='whether to fine-tune static pre-trained word vectors')
parser.add_argument('-multichannel', type=bool, default=True, help='whether to use 2 channel of word vectors')
# parser.add_argument('-pretrained-name', type=str, default='sgns.zhihu.word',
#                        help='filename of pre-trained word vectors')
parser.add_argument('-pretrained-path', type=str, default='pretrained', help='path of pre-trained word vectors')
parser.add_argument('-EMBEDDING_SIZE', type=int, default=200, help='number of embedding dimension [default: 128]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
# parser.add_argument('-early-stopping', type=int, default=1000,
#                        help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
# parser.add_argument('-test-interval', type=int, default=50,
#                    help='how many steps to wait before testing [default: 100]')
# parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
args = parser.parse_args()

Save_dir = './snapshot'
pretrained_name = 'sgns.zhihu.word'
test_interval = 10
# 语料路径
PATH_1 = './data/1.1房地产买卖合同.txt'
PATH_2 = './data/1.2一般商品买卖合同.txt'
PATH_3 = './data/1.3一般买卖合同（合同+订单）.txt'
PATH_4 = './data/1.4代理经销、分销合同.txt'
PATH_5 = './data/1.5超市与供应商购销合同.txt'
PATH_6 = './data/1.6采购、买卖配套文本.txt'
PATH_7 = './data/1.7特殊模式买卖合同.txt'

