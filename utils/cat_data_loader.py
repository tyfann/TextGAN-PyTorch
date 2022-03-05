# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : cat_data_loader.py
# @Time         : Created at 2019-05-31
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

<<<<<<< HEAD
import sys# 加了这了
sys.path.append("./")  ## 这个
=======
import sys
sys.path.append("../")
>>>>>>> catGAN add

import random
from torch.utils.data import Dataset, DataLoader

from utils.text_process import *


class GANDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

# 表示train中的内容
class CatGenDataIter:
    def __init__(self, samples_list, shuffle=None):
        self.batch_size = cfg.batch_size
        # self.batch_size = 64
        self.max_seq_len = cfg.max_seq_len
        # self.max_seq_len = 37
        self.start_letter = cfg.start_letter
        # self.start_letter = 1
        self.shuffle = cfg.data_shuffle if not shuffle else shuffle
        # self.shuffle = False if not shuffle else shuffle

        if cfg.if_real_data:
        # if True:
            self.word2idx_dict, self.idx2word_dict = load_dict(cfg.dataset)
            # self.word2idx_dict, self.idx2word_dict = load_dict('x')


        self.loader = DataLoader(
            dataset=GANDataset(self.__read_data__(samples_list)),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True)

        self.input = self._all_data_('input')
        self.target = self._all_data_('target')
        self.label = self._all_data_('label')  # from 0 to k-1, different from Discriminator label

    def __read_data__(self, samples_list):
        """
        input: same as target, but start with start_letter.
        """
        inp, target, label = self.load_data(samples_list)
        # inp, target, label = self.prepare(samples_list)
        all_data = [{'input': i, 'target': t, 'label': l} for (i, t, l) in zip(inp, target, label)]
        return all_data

    def random_batch(self):
        """Randomly choose a batch from loader, please note that the data should not be shuffled."""
        idx = random.randint(0, len(self.loader) - 1)
        return list(self.loader)[idx]

    def _all_data_(self, col):
        return torch.cat([data[col].unsqueeze(0) for data in self.loader.dataset.data], 0)

    def prepare(self, samples_list, gpu=False):
        """Add start_letter to samples as inp, target same as samples"""
<<<<<<< HEAD
        # all_samples = torch.cat(samples_list, dim=0).long()
        all_samples = samples_list.long()
=======
        # all_samples = samples_list   #获取文件中的全部内容
        # target = all_samples[::2][0]  #获取文件偶数（从0开始）行的标题
        # inp = torch.zeros(target.size()).long()
        # inp[:, 0] = 1
        # inp[:, 1:] = target[:, :37 - 1]
        # label = all_samples[1:2, 0]


        # num = len(samples_list)
        # print(num)
        # print(label)
        # for i in range(0,num,2):
        #      label[i] = samples_list[i][2][0]
        # print(label)
        # for idx in range(len(samples_list)):
        #     start = sum([samples_list[i].size(0) for i in range(idx)])
        #     label[start: start + samples_list[idx].size(0)] = idx

        # len(sample_list) == k_label
        all_samples = torch.cat(samples_list, dim=0).long()
>>>>>>> catGAN add
        target = all_samples
        inp = torch.zeros(all_samples.size()).long()
        inp[:, 0] = self.start_letter
        inp[:, 1:self.max_seq_len] = target[:, :self.max_seq_len - 1]

        label = torch.zeros(all_samples.size(0)).long()
        for idx in range(len(samples_list)):
            start = sum([samples_list[i].size(0) for i in range(idx)])
            label[start: start + samples_list[idx].size(0)] = idx

        # shuffle
        perm = torch.randperm(inp.size(0))
        inp = inp[perm].detach()
        target = target[perm].detach()
        label = label[perm].detach()

        if gpu:
            return inp.cuda(), target.cuda(), label.cuda()
        return inp, target, label

    def load_data(self, filename):
        """Load real data from local file"""
        self.tokens = get_tokenlized(filename)
        samples_index = tokens_to_tensor(self.tokens, self.word2idx_dict)
<<<<<<< HEAD
        print(samples_index)
=======
        # print(samples_index[0])
>>>>>>> catGAN add
        return self.prepare(samples_index)

# 表示testdata中的内容
class CatClasDataIter:
    """Classifier data loader, handle for multi label data"""

    def __init__(self, samples_list, given_target=None, shuffle=None):
        """
        - samples_list:  list of tensors, [label_0, label_1, ..., label_k]
        """
        self.batch_size = cfg.batch_size
        self.max_seq_len = cfg.max_seq_len
        self.start_letter = cfg.start_letter
        self.shuffle = cfg.data_shuffle if not shuffle else shuffle

        self.loader = DataLoader(
            dataset=GANDataset(self.__read_data__(samples_list, given_target)),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True)

        self.input = self._all_data_('input')
        self.target = self._all_data_('target')

    def __read_data__(self, samples_list, given_target=None):
        inp, target = self.prepare(samples_list, given_target)
        all_data = [{'input': i, 'target': t} for (i, t) in zip(inp, target)]
        return all_data

    def random_batch(self):
        idx = random.randint(0, len(self.loader) - 1)
        return list(self.loader)[idx]
        # return next(iter(self.loader))

    def _all_data_(self, col):
        return torch.cat([data[col].unsqueeze(0) for data in self.loader.dataset.data], 0)

    @staticmethod
    def prepare(samples_list, given_target=None, detach=True, gpu=False):
        """
        Build inp and target
        :param samples_list: list of tensors, [label_0, label_1, ..., label_k]
        :param given_target: given a target, len(samples_list) = 1
        :param detach: if detach input
        :param gpu: if use cuda
        :returns inp, target:
            - inp: sentences
            - target: label index, 0-label_0, 1-label_1, ..., k-label_k
        """
        if len(samples_list) == 1 and given_target is not None:
            inp = samples_list[0]
            if detach:
                inp = inp.detach()
            target = torch.LongTensor([given_target] * inp.size(0))
            if len(inp.size()) == 2:  # samples token, else samples onehot
                inp = inp.long()
        else:
            inp = torch.cat(samples_list, dim=0)  # !!!need .detach()
            if detach:
                inp = inp.detach()
            target = torch.zeros(inp.size(0)).long()
            if len(inp.size()) == 2:  # samples token, else samples onehot
                inp = inp.long()
            for idx in range(1, len(samples_list)):
                start = sum([samples_list[i].size(0) for i in range(idx)])
                target[start: start + samples_list[idx].size(0)] = idx

        # shuffle
        perm = torch.randperm(inp.size(0))
        inp = inp[perm]
        target = target[perm]

        if gpu:
            return inp.cuda(), target.cuda()
        return inp, target
<<<<<<< HEAD

if __name__ == '__main__':

    filename = '/root/autodl-tmp/TextGAN/dataset/x.txt'
    catGenDataIter = CatGenDataIter(filename)

    # t = torch.tensor([[1],[2],[0]])
    # train_samples_list = [t, t, t]
    # catGenDataIter = CatGenDataIter(train_samples_list)
    # samples = list(torch.tensor([[i for i in range(15)],[2 for i in range(15)]]))
    # catClasDataIter = CatClasDataIter(samples)
    # all_train_data = catGenDataIter.prepare(train_samples_list)
    print(catGenDataIter.loader.dataset.data)
=======
# if __name__ == '__main__':
#     t = torch.tensor([[1],[2],[3]])
#     train_samples_list = [t,t,t]
#     catGenDataIter = CatGenDataIter(train_samples_list)
    # samples = list(torch.tensor([[i for i in range(15)],[2 for i in range(15)]]))
    # catClasDataIter = CatClasDataIter(samples)
    # all_train_data = catGenDataIter.prepare(train_samples_list)
    # print(type(train_samples_list))
    # print(catGenDataIter.loader.dataset.data)
>>>>>>> catGAN add
