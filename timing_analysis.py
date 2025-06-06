'''
Modify the following variables in the `__main__` block as needed:
   * `all_sizes`: a list of window‐side lengths (e.g., `[16, 32, 64, 128]`). Each entry defines a square window of size `window_size × window_size` (so sequence length = `window_size²`).
   * `num_landmarks`: number of landmarks (default `16`).
   * `iters`: number of iterations for the Moore-Penrose pseudo-inverse (default `3`).
   * `device`: either `'cpu'` or `'cuda'` (default `'cpu'`).
'''

import torch
import numpy as np
from einops import rearrange
import torch.nn as nn
import math
import torch.nn.functional as F
import time


class ExpLinearAfterThreshold(nn.Module):
    def __init__(self, max_val=5.0):
        super(ExpLinearAfterThreshold, self).__init__()
        self.max_val = max_val

    def forward(self, x):
        return torch.exp(x.clamp(max=self.max_val)) + torch.relu(x - self.max_val)


activ_normal = nn.Softmax(dim=-1)
activ_new = ExpLinearAfterThreshold(max_val=5.0)
pooling = F.adaptive_avg_pool2d


def original_attention(q, k, v):
    '''
    Original Window Attention Mechanism
    '''
    attn = (q @ k.transpose(-2, -1))
    attn = activ_normal(attn)
    attn = attn @ v
    return attn


def moore_penrose_iter_pinv(x, iters = 6):
    '''
    Iterative Moore-Penrose Pseudo Inverse
    '''
    device = x.device
    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row) + 1e-15)

    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z


def closest_square_factors(N: int) -> tuple[int,int]:
    """
    Return a pair (r, c) with r*c = N and |r-c| minimized.
    """
    s = int(math.isqrt(N))   # floor of sqrt(N)
    for i in range(s, 0, -1):
        if N % i == 0:
            return i, N // i
        

def pnp_nystra_attention(q, k, v, window_size, num_landmarks = 16, iters = 3):
    B_, num_heads, N, dim = q.shape
    h, w = closest_square_factors(num_landmarks)

    q_m = pooling(q.reshape(B_*num_heads, window_size, window_size, dim).permute(0, 3, 1, 2), output_size = (h, w)).permute(0, 2, 3, 1).reshape(B_, num_heads, num_landmarks, dim)
    k_m = pooling(k.reshape(B_*num_heads, window_size, window_size, dim).permute(0, 3, 1, 2), output_size = (h, w)).permute(0, 2, 3, 1).reshape(B_, num_heads, num_landmarks, dim)

    temp = activ_new(q_m @ k_m.transpose(-2, -1))

    pseudo_inv = moore_penrose_iter_pinv(temp, iters)
    
    prod = (activ_new(q @ k_m.transpose(-2, -1)) @ pseudo_inv) @ (activ_new(q_m @ k.transpose(-2,-1)) @ torch.cat([v, torch.ones_like(v[..., :1])], dim=-1))

    return (prod[..., :-1] / (prod[..., -1].unsqueeze(-1) + 1e-12))

if __name__ == '__main__':

    '''
    Script used for the ablation study on the effect of number of tokens on attention forward pass time
    of the pnp-nystra mechanism as opposed to the original window attention mechanism. 

    (NOT FOR FULL MODEL TIMES REPORTED IN EXPERIMENTS SECTION OF PAPER)
    '''

    # choose device for time comparison

    # device = 'cuda'
    device = 'cpu'

    seed = 32
    torch.manual_seed(seed)

    all_sizes = [16, 32, 64, 128] # window sizes to test (sequence length = window_size * window_size)
    times_orig = []
    times_pnp = []

    for window_size in all_sizes:
        tokens = window_size * window_size

        print(f"Testing window size {window_size} with {tokens} tokens")
        
        q, k, v = torch.rand(1, 8, tokens, 32).to(device), torch.rand(1, 8, tokens, 32).to(device), torch.rand(1, 8, tokens, 32).to(device)

        att_normal = original_attention(q, k, v)

        # change landmarks and iterations for pnp-nystra attention
        att_approx = pnp_nystra_attention(q, k, v, window_size, num_landmarks=16, iters=3)

        error = torch.nn.MSELoss()(att_normal, att_approx)

        print(f"Error between attention matrices = {error}")

        times_new = []

        for iteration in range(100):
            t1 = time.time()
            _ = pnp_nystra_attention(q, k, v, window_size)
            times_new.append((time.time()-t1)*1000)
        times_pnp.append(np.mean(times_new[5:]))
        print('PnP-Nystra takes (ms) : {:.2f}'.format(np.mean(times_new[5:])))

        times = []

        for iteration in range(100):
            t1 = time.time()
            _ = original_attention(q, k, v)
            times.append((time.time()-t1)*1000)
        times_orig.append(np.mean(times[5:]))
        print('Original takes (ms) : {:.2f}'.format(np.mean(times[5:])))


        print("---------------------------------------------------------")
