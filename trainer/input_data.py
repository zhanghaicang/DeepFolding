import pandas as pd
import random
import numpy as np
import gc 
from util import RunMode

class MemDataset:
    pass

class TfRecordDataset:
    def __init__(self, train_file_prefix='', chunk_num=0, val_size = 1,\
            test_file_prefix='', unlabel_file_prefix='', unlabel_chunk_num=0):
        self.chunk_num = chunk_num
        self.train_file_prefix = train_file_prefix
        self.test_file_prefix = test_file_prefix
        self.unlabel_file_prefix = unlabel_file_prefix
        self.unlabel_chunk_num = unlabel_chunk_num

        idx = range(chunk_num)
        #random.shuffle(idx)
        if type(val_size) == float:
            train_chunk_num = np.ceil(chunk_num * (1.0 - val_size))
        else:
            train_chunk_num = chunk_num - val_size

        self.train_chunks = idx[:train_chunk_num]
        self.val_chunks = idx[train_chunk_num:]

    def get_chunks(self, mode):
        input_file_prefix = self.train_file_prefix
        if mode == RunMode.TRAIN:
            chunks = self.train_chunks
            random.shuffle(chunks)
        elif mode == RunMode.VALIDATE:
            chunks = self.val_chunks
        elif mode == RunMode.TEST:
            input_file_prefix = self.test_file_prefix
            chunks = [0]
        elif model == RunMode.UNLABEL:
            input_file_prefix = self.unlabel_file_prefix
            chunks = range(self.unlabel_chunk_num)
            random.shuffle(chunks)
        return ['{}_{:d}.tfrecord'.format(input_file_prefix, c)\
                for c in chunks]
