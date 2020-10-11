## FIXME: 数据处理  
import numpy as np
import pandas as pd
import scanpy as sc
import anndata

def normalize(data: pd.DataFrame, data_T: pd.DataFrame):
    """
    输入：
        data:  drop数据集
        data_T: 真实数据集
    返回：
        1.预处理后的数据集
        2.相应维度的真实数据集
    """
    ## TODO: 将 3000x500 → 500x3000
    # data = data.transpose()
    # data_T = data_T.transpose()

    adata = anndata.AnnData(data)   ## 将 pandas数据 实例化为 anndata类型

    ## 1. Gene filter 基因过滤
    sc.pp.filter_genes(adata, min_counts=1)   # 500x【3000】 → 500x【1346】 
    sc.pp.filter_cells(adata, min_counts=1)

    # 缺失数据集 drop被过滤成 500x1344，真实数据集 true也转成相应的行列
    adata_gene_names = adata.var_names    ## (1346, )   
    adata_cell_names = adata.obs_names   ## (500, )    
    adata_gene_idx   = list(range(len(adata_gene_names)))    
    adata_gene_dict  = dict(zip(adata_gene_names, adata_gene_idx))  # 建立drop过滤后的 基因名称 —— 所在下标 之间的联系
 
    dataT_gene_names = data_T.columns.to_list()   ## (3000, ) 
    dataT_gene_idx  = list(range(len(dataT_gene_names)))     
    dataT_gene_dict  = dict(zip(dataT_gene_names, dataT_gene_idx))

    ## (1346, 500) 这里故意初始化转置矩阵，为了方便后面赋值    (1985,1000)
    data_true_T = np.zeros((adata_gene_names.shape[0], adata_cell_names.shape[0]))  


    for i in adata_gene_names:
        ## 将基因名i在原数据集所在行复制到基因名i在过滤后新数据集所在的行的位置中（相当于一个映射）
        ## adata_gene_dict[i]可理解为取基因名i在adata中所在的行下标；
        ## dataT_gene_dict[i]可理解为取基因名i在原始(转置)数据集中所在的行下标
        data_true_T[adata_gene_dict[i]] = data_T.values.T[dataT_gene_dict[i]]   
        # could not broadcast input array from shape (3000) into shape (1346)
    
    ## 2. Normalization 归一化 
    for i in list(range(adata.shape[0])):
        total = adata.obs.n_counts[i]
        median = np.median(adata.obs.n_counts)
        for j in list(range(adata.shape[1])):
            adata.X[i][j] = (adata.X[i][j] / total) * median  

    data_norm = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    # data_norm.to_csv('data/drop_500x3000/normalize.csv')


    ## 3. Log Transformation 取对数
    sc.pp.log1p(adata, base=2)  ## FIXME: 【试性能、优化】 base指定底数 

    data_log = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    # data_log.to_csv('data/drop_500x3000/log.csv')

    return adata.X, adata.obs_names, adata.var_names, data_true_T.T
    # return adata.X.T, adata.var_names, adata.obs_names, data_true_T  TODO:3000x500






        






'''
def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1) ## (3000,500)  
        sc.pp.filter_cells(adata, min_counts=1) ## (1345,500)https://scanpy.readthedocs.io/en/stable/api/scanpy.pp.filter_genes.html?highlight=filter_genes

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)  ## TODO: 对数据归一化：
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)  ## TODO: 对数化的数据(基因表达值)，方便可视化画图分析

    if normalize_input:
        sc.pp.scale(adata)

    return adata
'''