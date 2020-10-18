import numpy as np

def get_batch(X, X_, Adj, size):   
    a = np.random.choice(len(X), size, replace=False)
    b = Adj[a] 
    return X[a], X_[a], (b.T[a]).T  


def noise_validator(noise, allowed_noises):   
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
