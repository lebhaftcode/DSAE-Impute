import os
import pandas as pd
import numpy as np
from scipy.stats import expon


def noise(noise: str, x):   
 
    if noise == 'gaussian':
        n = np.random.normal(0, 0.1, (len(x), len(x[0])))
        return x+n

    if 'mask' in noise:  
        frac = float(noise.split('-')[1])
        temp = np.copy(x)
        for i in temp:
            n = np.random.choice(len(i), round(frac * len(i)), replace=False)
            i[n] = 0
        return temp  

    if noise == 'sp':
        pass
