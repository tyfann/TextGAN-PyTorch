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
    file = open(datapath + 'toutiao_cat_data.txt', 'r', encoding='utf-8')
    line = file.readline()  # 调用文件的 readline()方法
    # list2 = [0]*15
    Name = 'x'
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
        # list2[count] = list2[count] + 1
        num = str(num)

        if len(lists) > 1:
            # if (list2[count] < 401):
            aim = Name + '_cat' + num + '.txt'
            f = open(datapath + aim, 'a+', encoding='utf-8')
            f.writelines(lists[3]+ '\n')
            f.close()
        line = file.readline()
    file.close()
    for i in range(15):
        name = Name + "_cat" + str(i)
        aim = name+'.txt'
        fname = open(datapath + aim, "rb")
        cut(name, fname, train_ratio = 0.8)
        fname.close()
    train = open(datapath + Name + '.txt', 'a+', encoding='utf-8')
    test = open(testpath + Name + '_test.txt', 'a+', encoding='utf-8')
    for i in range(15):
        name = Name + "_cat" + str(i)
        aim = name + '.txt'
        goal = name + '_test.txt'
        train_data = open(datapath + aim, 'r', encoding='utf-8')
        test_data = open(testpath + goal, 'r', encoding='utf-8')
        line1 = train_data.readline()
        line2 = test_data.readline()
        # count1 = 0
        # count2 = 0
        while line1:
            train.writelines(line1)
            line1 = train_data.readline()
            # count1 = count1 + 1
        train_data.close()
        while line2:
            test.writelines(line2)
            line2 = test_data.readline()
            # count2 = count2 + 1
        test_data.close()
        # print("range" + str(i) + "  " + str(count1) + "  " + str(count2))
    train.close()
    test.close()


