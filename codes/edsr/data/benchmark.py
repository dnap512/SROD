import os

from data import common
from data import srdata

import numpy as np

import torch
import torch.utils.data as data

class Benchmark(srdata.SRData):
    def __init__(self, args, name='', train=True, benchmark=True):
        self.degrad = args.degrad
        self.png = args.png
        super(Benchmark, self).__init__(
            args, name=name, train=train, benchmark=True
        )
        

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'benchmark', self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        if self.input_large:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubicL_{}'.format(self.degrad))
        else:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubic_{}'.format(self.degrad))
        
        if self.png=='png':
            self.ext = ('', '.png')
        else:
            self.ext = ('', '.jpg')

