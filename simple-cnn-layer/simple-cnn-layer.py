import numpy as np

def conv2d(x, W, b):
    N, C_in, H, W_in = x.shape
    C_out, _, KH, KW = W.shape
    H_out = H - KH + 1
    W_out = W_in - KW + 1

    # Build sliding windows via stride tricks: (N, C_in, H_out, W_out, KH, KW)
    s = x.strides
    windows = np.lib.stride_tricks.as_strided(
        x,
        shape=(N, C_in, H_out, W_out, KH, KW),
        strides=(s[0], s[1], s[2], s[3], s[2], s[3])
    )

    # Einsum: contract over C_in, KH, KW
    return np.einsum('nchwij,ocij->nohw', windows, W) + b[None, :, None, None]