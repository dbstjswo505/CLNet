import numpy as np
import torch
import pdb

sparsity_mask = torch.zeros((900,900), dtype=torch.long)
stride_ed = 3
stride_st = 3

for i in range(900):
    for j in range(900):
        if i % stride_st == 0:
            if j % stride_ed == 0: 
                sparsity_mask[i][j] = 1
            else: 
                z=1

causal_mask = torch.triu(torch.ones(900, 900))
np.save('sparsity_mask.npy', sparsity_mask.numpy())
np.save('causal_mask.npy', causal_mask.numpy())


