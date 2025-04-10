import numpy as np
import torch

x = np.array([[1, 2], [3, 4]])
t = torch.from_numpy(x)
print("NumPy works ✅", t)