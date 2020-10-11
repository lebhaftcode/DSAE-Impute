''' 
TODO: 对原始数据集随机遮盖

      掩码数组也是一种数组，它本质上是对数组的一种特殊处理
      掩码的目的就是表明被掩数据的某些位的作用的，通常用来表明被掩数据哪些要处理，哪些不必处理。
'''
import os
import pandas as pd
import numpy as np
from scipy.stats import expon
from matplotlib.pyplot import MultipleLocator

class MaskedArray(object):

    def __init__(self, data=None, mask=None, distr="exp", dropout=0.01, seed=1):
        self.data = np.array(data)     # 格式化为np数组
        self._binMask = np.array(mask) # 格式化为np数组
        self.shape = data.shape
        self.distr = distr
        self.dropout = dropout
        self.seed = seed

    @property
    def binMask(self):
        return self._binMask

    @binMask.setter
    def binMask(self, value):
        self._binMask = value.astype(bool)  # astype:数据格式转换

    # TODO: 由data和 binMask 获得掩码数组
    def getMaskedMatrix(self):  
        print(self._binMask)
        maskedMatrix = self.data.copy()
        maskedMatrix[~self.binMask] = 0    # 若 binMask中为 True(1)，则原数据中对应值无效 0(即被掩盖)
        return maskedMatrix    

    # TODO: 获取掩码后数据，非定长矩阵，缺失处不补0
    def getMasked(self, rows=True):  
        """ Generator for row or column mask """
        compt = 0
        if rows:    # TODO: 选取由行对数据集做掩码
            while compt < self.shape[0]:
            # yield: Python中的（generator）生成器并不会一次返回所有结果，
            # 而是每次遇到yield关键字后返回相应结果，并保留函数当前运行状态，等待下一次调用。
                yield [ 
                    self.data[compt, idx]
                    for idx in range(self.shape[1])
                    if not self.binMask[compt, idx]
                ]
                compt += 1
        else:    # TODO: 选取由行对数据集做掩码
            while compt < self.shape[1]:
                yield [
                    self.data[idx, compt]
                    for idx in range(self.shape[0])
                    if not self.binMask[idx, compt]
                ]
                compt += 1

    # TODO: 获取输出被掩盖处原始值
    def getMasked_flat(self):
        print(self.data[~self.binMask]) #######
        return self.data[~self.binMask]

    # TODO: 便于修改一些数据
    def copy(self):
        args = {"data": self.data.copy(), "mask": self.binMask.copy()}
        MaskedArray(**args)
        return MaskedArray(**args)

    def get_probs(self, vec):
        return {
            "exp": expon.pdf(vec, 0, 20),
            "uniform": np.tile([1. / len(vec)], len(vec)),
        }.get(self.distr)

    def get_Nmasked(self, idx):
        cells_g = self.data[:, idx]
        dp_i = (1 + (cells_g == 0).sum() * 1.) / self.shape[0]
        dp_f = np.exp(-2 * np.log10(cells_g.mean()) ** 2)
        return 1 + int((cells_g == 0).sum() * dp_f / dp_i)


    # TODO: 由掩盖概率生成binMask
    def generate(self):
        np.random.seed(self.seed)
        self.binMask = np.ones(self.shape).astype(bool)  # 开始将 binMask设为全1矩阵

        for c in range(self.shape[0]):
            cells_c = self.data[c, :]
            # Retrieve indices of positive values 检索正值索引
            ind_pos = np.arange(self.shape[1])[cells_c > 0]
            cells_c_pos = cells_c[ind_pos]
            # Get masking probability of each value 获取每个值的掩盖概率

            if cells_c_pos.size > 5:
                probs = self.get_probs(cells_c_pos)
                n_masked = 1 + int(self.dropout * len(cells_c_pos))
                if n_masked >= cells_c_pos.size:
                    print("Warning: too many cells masked for gene {} ({}/{})".format(c, n_masked, cells_c_pos.size))
                    n_masked = 1 + int(0.5 * cells_c_pos.size)

                masked_idx = np.random.choice(cells_c_pos.size, n_masked, p=probs / probs.sum(), replace=False)
                self.binMask[c, ind_pos[sorted(masked_idx)]] = False

    
def main(data):
    row = data.index   # pandas数组可直接取行列索引，但umpy不可，只含数据值
    col = data.columns

    maskedData = MaskedArray(data=data)   # 实例化一个类对象
    # print(maskedData) 
    # print(maskedData.binMask)

    maskedData.generate()
    # print(maskedData.binMask)
    # pd.DataFrame(maskedData.binMask, row, col).to_csv('data/mni_500x3000/binMask.csv')

    maskedMatrix = maskedData.getMaskedMatrix()
    drop = pd.DataFrame(maskedMatrix, row, col)
    # drop.to_csv('data/neuron9k/raw_500x3000.csv')          ### TODO: 调用进行随机掩盖
    
    return drop