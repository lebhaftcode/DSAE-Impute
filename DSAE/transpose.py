import pandas as pd

## 在Vscode中./代表打开的文件夹所在路径
a = pd.read_csv('data/sim_2000x4000/true.csv', index_col=0) 
b = a.transpose()
b.to_csv('data/sim_2000x4000/true_T.csv')

