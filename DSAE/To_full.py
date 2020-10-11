import numpy as np 
import pandas as pd

def  getAll(csv_impute, csv_raw):

    ## TODO: 将 3000x500 → 500x3000
    # csv_impute = csv_impute.transpose()
    # csv_raw = csv_raw.transpose()

    csv_full = csv_raw.astype(np.float32)   ### 注意数据类型
    columns_all = csv_full.columns.values
    columns_part = csv_impute.columns.values

    for index_0, value_0 in enumerate(columns_part):
        for index, value in enumerate(columns_all):  # enumerate可在获取列表值的同时获取下标
            if( value_0 == value):
                csv_full.values[:, index] = csv_impute.values[:, index_0]   ## TODO: 替换补全前矩阵的整列值

    # print(csv_full)

    csv_final = impute_zero_only(csv_raw, csv_full)
    csv_final = np.abs(csv_final)    ## 将补全后的 负值 → 绝对值

    # print(csv_final)
    # csv_true.to_csv('data/sim_2000x4000/impute_2000x4000.csv')

    return csv_final
    # return csv_full


def impute_zero_only(csv_raw, csv_full):   ### TODO: 只保留列中 0值补全结果、非 0值直接用原值覆盖
    row, col = np.nonzero(csv_raw.values)  ##? 借鉴
    data = []
    for i in range(len(row)):
        csv_full.values[row[i]][col[i]] = csv_raw.values[row[i]][col[i]]
        data.append(csv_raw.values[row[i]][col[i]])
    
    return csv_full


    