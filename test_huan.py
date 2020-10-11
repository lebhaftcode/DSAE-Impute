from deepautoencoder import StackedAutoEncoder
import deepautoencoder.Pre_process as Pre_process
import deepautoencoder.To_full as To_full
import deepautoencoder.Dropout as Dropout
import pandas as pd
import numpy as np 
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import time
from sklearn.metrics.pairwise import cosine_similarity

##? TODO: 加了几行代码，便于外部调用
# print('[[test_huan]]', flush=True)
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--inputT', type=str, default='data/drop_500x3000/true.csv')
# parser.add_argument('--inputR', type=str, default='data/drop_500x3000/raw.csv')
# args = parser.parse_args() 

# data_T = pd.read_csv(args.inputT, index_col=0)  ## 真实矩阵
# # data_raw = Dropout.main(data_T)
# data_raw = pd.read_csv(args.inputR, index_col=0)

# 1.数据加载
# start = time.time()
# data_T = pd.read_csv('data/68k PBMCs/true_500x3000.csv', index_col=0)  ## 真实矩阵
# # data_raw = Dropout.main(data_T)
# data_raw = pd.read_csv('data/68k PBMCs/raw_500x3000.csv', index_col=0)   ## 缺失矩阵

data_T = pd.read_csv('data/sim_2000x4000/true.csv', index_col=0)  ## 真实矩阵
# data_raw = Dropout.main(data_T)
data_raw = pd.read_csv('data/sim_2000x4000/raw.csv', index_col=0)   ## 缺失矩阵

#? TODO: 相似度矩阵 
adj = cosine_similarity(data_raw.values)
print(adj) #mua

# 2.数据预处理：pandas(输入) → anndata → numpy(输出)
data_raw_process, row, col, data_true_part = Pre_process.normalize(data_raw, data_T)  # 500x3000 → 500x1344


## TODO: 随机选取 80% 作为训练集   
# idx = np.random.rand(data_true.shape[0]) < 0.8   
# drop = data_raw_process[idx] 
# true = data_true[idx]


# 3.模型定义
model = StackedAutoEncoder(dims = [600, 256],  #!【注意】这里不再是数组里有几个元素就堆叠几层，而是方括号中元素个数要与模型层数相对应（即模型的堆叠层数是写死的，这里要与写死的层个数匹配）
                           activations = ['sigmoid', 'relu'],
                           epoch = [3000, 1000], 
                           loss = 'rmse',
                           lr = 4e-3,
                           noise = None,   ## 默认 noise = None，不需加''
                           batch_size = 64, 
                           print_step = 200,
                           Adj = adj)   ## 每隔200轮输出一次
# 4.模型训练
model.fit(data_raw_process, data_true_part)   ## 训练模型
predict = model.predict(data_raw_process)     # TODO: predict 等价于 transform + decode

impute_part = pd.DataFrame(predict, index=row, columns=col)

impute = To_full.getAll(impute_part, data_raw)  ## (500, 1346) → (500, 3000) & 负值 → 绝对值
# end = time.time()
# print('耗时：{}'.format(end-start))
# print(impute)
 
# impute.values[np.where(impute.values < 0.)] = 0.0  ## 将补全后的 负值 → 0
# impute = np.abs(impute)  ## 将补全后的 负值 → 绝对值

impute.to_csv('data/sim_2000x4000/My-a2_impute.csv')

## 看 weight和 bias

# 5.模型评估  TODO: 3000x500的话先 data_T.transpose()
print("------------------------- The metrics of this {}x{}--------------------------- ".format(data_T.values.shape[0], data_T.values.shape[1]))
print("绝对误差 MAE = {0:.3f}".format( mean_absolute_error(data_T, impute) ))
print("归一化均方误差 NMSE = {0:.3f}".format( (np.linalg.norm(data_T - impute) / np.linalg.norm(data_T)) ))
print("均方误差 RMSE = {0:.3f}".format( mean_squared_error(data_T, impute) ** 0.5 ))
print("皮尔逊相关系数 PCC = {0:.3f}".format( pearsonr(data_true_part.reshape(-1), impute_part.values.reshape(-1))[0] ))


