import torch.nn.functional as F
import numpy as np

img = np.random.randn(3,5)

coord_x = np.tile(range(5), (3, 1)) / (H/2) - 1
coord_y = np.tile(range(W), (H, 1)).T / (W/2) - 1

#print(coord_x.shape, coord_y.shape)

grid = np.stack([coord_x, coord_y])
grid = np.transpose(grid, [1,2,0])
grid = np.stack([grid] * num_pairs)