import numpy as np

def softmax(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        e = np.exp(x - x.max())
        return e / e.sum()
    else:
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)