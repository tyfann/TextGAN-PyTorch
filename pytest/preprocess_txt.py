# -*- encoding: utf-8 -*-
# @ModuleName: preprocess_txt
# @Function: 
# @Author: Yufan-tyf
# @Time: 2022/3/4 9:40 PM

import random

datapath = '../dataset/'
testpath = '../dataset/testdata/'

def cut(name,fname, train_ratio):
    lines = fname.readlines()
    n_total = len(lines)
    aim = name+'.txt'
    goal = name+'_test.txt'
    train_offset = int(n_total * train_ratio)
    random.shuffle(fname.read())
    train_data = open(datapath + aim, 'wb')
    test_data = open(testpath + goal, 'wb')


    for i, line in enumerate(lines):
        if i < train_offset:
            train_data.write(line)
        else:
            test_data.write(line)

    train_data.close()
    test_data.close()


if __name__ == "__main__":
    file = open(datapath + 'x.txt', 'r', encoding='utf-8')
    line = file.readline()  # 调用文件的 readline()方法
    while line:
        str1, str2 = line.split('\t', 1)
        str2 = str2.split('\n')
        num = int(str2[0])
        if len(str1) > 1:
            aim = 'x_cat' + str2[0] + '.txt'
            f = open(datapath + aim, 'a+', encoding='utf-8')
            f.writelines(str1 + '\n')
            f.close()
        line = file.readline()
    file.close()
    for i in range(15):
        name = "x_cat" + str(i)
        aim = name+'.txt'
        fname = open(datapath + aim, "rb")
        cut(name, fname, train_ratio = 0.8)
        fname.close()
