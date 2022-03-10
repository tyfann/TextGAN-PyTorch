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
    train = open(datapath + 'x.txt', 'ab+')
    test = open(testpath + 'x_test.txt', 'ab+')

    for i, line in enumerate(lines):
        if i < train_offset:
            train_data.write(line)
            train.write(line)
        else:
            test_data.write(line)
            test.write(line)

    train_data.close()
    test_data.close()
    train.close()
    test.close()

if __name__ == "__main__":
    file = open(datapath + 'toutiao_cat_data.txt', 'r', encoding='utf-8')
    line = file.readline()  # 调用文件的 readline()方法
    while line:
        lists = line.split("_!_")
        num = int(lists[1])
        if (num > 105 and num < 111):
            num = num - 101
        elif (num > 111):
            num = num - 102
        else:
            num = num - 100
        count = num
        num = str(num)

        if len(lists) > 1:
            aim = 'x_cat' + num + '.txt'
            f = open(datapath + aim, 'a+', encoding='utf-8')
            f.writelines(lists[3]+ '\n')
            f.close()
            F = open(datapath + 'x.txt', 'a+', encoding='utf-8')
            F.writelines(lists[3]+ '\n')
            F.close()
        line = file.readline()
    file.close()
    for i in range(15):
        name = "x_cat" + str(i)
        aim = name+'.txt'
        fname = open(datapath + aim, "rb")
        cut(name, fname, train_ratio = 0.8)
        fname.close()
