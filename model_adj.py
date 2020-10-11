import numpy as np
import matplotlib.pyplot as plt
Mat = np.random.randint(0,10,(64,64)) ## 生成64维，值在0-9之间的矩阵

plt.matshow(Mat,cmap=plt.cm.Blues)  ## spring
plt.show()