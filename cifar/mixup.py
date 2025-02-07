import numpy as np
import torch

def mixup_data(x, y, device, alpha = 1.0):
  if alpha > 0:
    lam = np.random.beta(alpha, alpha)
  else:
    lam = 1

  batch_size = x.shape[0]
  index_sample = torch.randperm(batch_size).to(device)

  mixed_x = lam * x + (1 - lam) * x[index_sample, :]
  y, y_sampled =  y, y[index_sample]

  return mixed_x, y, y_sampled, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
  return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
