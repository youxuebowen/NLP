#!/usr/bin/python
# -*-coding:utf-8-*-
# @Time:2021/10/30  20:56
# @Author:Mr.Wan
# @Email:2215225145@qq.com
# @File:Files.py
# Software:PyCharm
import os


def get_list(dir_path, save_patch):
    # 获得文件名
    dir_files = os.listdir(dir_path)
    # 转移到本项目所在文件夹
    os.chdir(save_patch)
    # 后期可以优化更改文件名
    with open("1.4代理经销、分销合同.txt", 'w', encoding='utf-8') as fp:
        for i in range(len(dir_files)):
            fp.write(dir_files[i] + '\n')
    # fp = open("dir_files.txt", 'w')
    # fp.write(dir_files)


if __name__ == "__main__":
    dir_patch = r"C:\Users\Administrator\Desktop\报告\文本分类\分类训练样例\1.4代理经销、分销合同"
    # dir_patch = r"C:\Users\Administrator\Desktop\报告\文本分类\分类训练样例\1.2一般商品买卖合同"
    # 获得当前位置
    save_patch = os.getcwd()
    get_list(dir_patch, save_patch)
    # 还有一个问题是，如何去除后缀，可以放在后面处理
    # 下一步是进一步生成可判别的文件，文件名称带标签，神经网络可处理，封装成一个类，可以随时调用生成目标文件，目标数据

