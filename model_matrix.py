import scanpy as sc
import numpy as np
import pandas as pd
import anndata
from matplotlib import rcParams
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
import seaborn as sns

data = pd.read_csv('./data/model_true.csv',index_col=0)
confusion_mtx = data.values

# confusion_mtx = np.array([[2,0,1,2],
#                           [1,2,0,3],
#                           [0,0,1,4]])

plt.matshow(confusion_mtx, cmap=plt.cm.pink)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('Label')
plt.xlabel('Pred')
plt.show()