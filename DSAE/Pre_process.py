import numpy as np
import pandas as pd
import scanpy as sc
import anndata

def normalize(data: pd.DataFrame, data_T: pd.DataFrame):
    adata = anndata.AnnData(data)   

    ## 1. Gene filter 
    sc.pp.filter_genes(adata, min_counts=1)   
    sc.pp.filter_cells(adata, min_counts=1)

    adata_gene_names = adata.var_names       
    adata_cell_names = adata.obs_names      
    adata_gene_idx   = list(range(len(adata_gene_names)))    
    adata_gene_dict  = dict(zip(adata_gene_names, adata_gene_idx)) 
 
    dataT_gene_names = data_T.columns.to_list()   
    dataT_gene_idx  = list(range(len(dataT_gene_names)))     
    dataT_gene_dict  = dict(zip(dataT_gene_names, dataT_gene_idx))

    data_true_T = np.zeros((adata_gene_names.shape[0], adata_cell_names.shape[0]))  

    for i in adata_gene_names:
        data_true_T[adata_gene_dict[i]] = data_T.values.T[dataT_gene_dict[i]]   
       
    
    ## 2. Normalization 
    for i in list(range(adata.shape[0])):
        total = adata.obs.n_counts[i]
        median = np.median(adata.obs.n_counts)
        for j in list(range(adata.shape[1])):
            adata.X[i][j] = (adata.X[i][j] / total) * median  

    data_norm = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)

    ## 3. Log Transformation 
    sc.pp.log1p(adata, base=2) 
    data_log = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)

    return adata.X, adata.obs_names, adata.var_names, data_true_T.T
