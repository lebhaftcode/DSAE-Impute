from deepautoencoder import StackedAutoEncoder
import deepautoencoder.Pre_process as Pre_process
import deepautoencoder.To_full as To_full
import deepautoencoder.Dropout as Dropout
import pandas as pd
import numpy as np 
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

# 1.数据加载
# data_T = pd.read_csv('data/drop_500x3000/true.csv', index_col=0)  ## 真实矩阵
# # data_raw = Dropout.main(data_T)
# data_raw = pd.read_csv('data/drop_500x3000/raw.csv', index_col=0)

data_T = pd.read_csv('data/sim_2000x4000/true_2000x4000.csv', index_col=0)  ## 真实矩阵
# data_raw = Dropout.main(data_T)
data_raw = pd.read_csv('data/sim_2000x4000/raw_2000x4000.csv', index_col=0)   ## 缺失矩阵


# 2.数据预处理：pandas(输入) → anndata → numpy(输出)
data_raw_process, row, col, data_true_part = Pre_process.normalize(data_raw, data_T)  # 500x3000 → 500x1344


## TODO: 随机选取 80% 作为训练集   
# idx = np.random.rand(data_true.shape[0]) < 0.8   
# drop = data_raw_process[idx] 
# true = data_true[idx]


# 3.模型定义
model = StackedAutoEncoder(dims = [400], 
                           activations = ['sigmoid'],   
                           epoch = [2000], 
                           loss = 'rmse', 
                           lr = 4e-3,
                           noise = 'mask-0.3',   ## 默认 noise=None，
                           batch_size = 64, 
                           print_step = 200)   ## 每隔200轮输出一次
# 4.模型训练
model.fit(data_raw_process, data_true_part)   ## 训练模型

# encode = model.transform(data_raw_process)  ## 自动编码器训练好后 对data_raw编码预测 (500, 3000) ---> (500, 200)
# predict = model.decode_(encode)       ## (500, 200)  ---> (500, 1344)
predict = model.predict(data_raw_process)     # TODO: predict 等价于 transform + decode
impute_part = pd.DataFrame(predict, index=row, columns=col)

impute = To_full.getAll(impute_part, data_raw)  ## (500, 1344)  ---> (500, 3000)
print(impute)
impute.values[np.where(impute.values < 0.)] = 0.0  ## 将补全后的 负值 → 0
print(impute)
impute.to_csv('data/sim_2000x4000/My1_impute.csv')

## 看 weight和 bias

# 5.模型评估
print("------------------------- The metrics of this 500x3000--------------------------- ")
print("绝对误差 MAE = {0:.3f}".format( mean_absolute_error(data_T, impute) ))
print("归一化均方误差 NMSE = {0:.3f}".format( (np.linalg.norm(data_T - impute) / np.linalg.norm(data_T)) ))
print("均方误差 RMSE = {0:.3f}".format( mean_squared_error(data_T, impute) ** 0.5 ))

# print("------------------------- The metrics of this 500x1344--------------------------- ")
# print("绝对误差 MAE = {0}".format( mean_absolute_error(data_true_part, impute_part.values) ))
# print("归一化均方误差 NMSE = {0}".format( (np.linalg.norm(data_true_part - impute_part.values) / np.linalg.norm(data_true_part)) ))
# print("均方误差 RMSE = {0}".format( mean_squared_error(data_true_part, impute_part.values) ** 0.5 ))

# PCC = pearsonr(data_true_part.reshape(-1), impute_part.values.reshape(-1))[0]
print("皮尔逊相关系数 PCC = {0:.3f}".format( pearsonr(data_true_part.reshape(-1), impute_part.values.reshape(-1))[0] ))

print()  ## 宝宝我好喜欢你..