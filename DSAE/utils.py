import numpy as np


def get_batch(X, X_, Adj, size):   
    '''  np.random.choice(a, size, replace, p) 其作用是按要求生成一个一维数组
         a是生成一维数组的来源，可以是int类型，可以是数组，也可以是list
         size 为从a中抽取的个数，即生成数组的维度
         replace 表示从a中是否不重复抽取，默认可重复
         p 给出抽取概率，默认随机
    '''
    a = np.random.choice(len(X), size, replace=False)
    b = Adj[a] 
    return X[a], X_[a], (b.T[a]).T  ## 'NoneType' object is not subscriptable


def noise_validator(noise, allowed_noises):   # 噪音合法验证
    '''Validates the noise provided'''
    try:
        if noise in allowed_noises:
            return True
        elif noise.split('-')[0] == 'mask' and float(noise.split('-')[1]):
            t = float(noise.split('-')[1])
            if t >= 0.0 and t <= 1.0:
                return True
            else:
                return False
    except:
        return False
    pass
