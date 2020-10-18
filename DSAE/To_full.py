import numpy as np 
import pandas as pd

def  getAll(csv_impute, csv_raw):
    csv_full = csv_raw.astype(np.float32)   
    columns_all = csv_full.columns.values
    columns_part = csv_impute.columns.values

    for index_0, value_0 in enumerate(columns_part):
        for index, value in enumerate(columns_all):  
            if( value_0 == value):
                csv_full.values[:, index] = csv_impute.values[:, index_0]   

    csv_final = impute_zero_only(csv_raw, csv_full)
    csv_final = np.abs(csv_final)   
     
    return csv_final
    

def impute_zero_only(csv_raw, csv_full):  
    row, col = np.nonzero(csv_raw.values)  
    data = []
    for i in range(len(row)):
        csv_full.values[row[i]][col[i]] = csv_raw.values[row[i]][col[i]]
        data.append(csv_raw.values[row[i]][col[i]])
    
    return csv_full
