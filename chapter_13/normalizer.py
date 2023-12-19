"""Components for normalize tensor."""
import numpy as np
import torch


class TorchRunningMeanStd:
    def __init__(self, shape=(), device='cpu'):
        self.device = device
        self.mean = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.var = torch.ones(shape, dtype=torch.float32, device=self.device)
        self.count = 0

    @torch.no_grad()
    def update(self, x):
        x = x.to(self.device)
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)

        # update count and moments
        n = x.shape[0]
        self.count += n
        delta = batch_mean - self.mean
        self.mean += delta * n / self.count
        m_a = self.var * (self.count - n)
        m_b = batch_var * n
        M2 = m_a + m_b + torch.square(delta) * self.count * n / self.count
        self.var = M2 / self.count

    @torch.no_grad()
    def normalize(self, x):
        return (x.to(self.device) - self.mean) / torch.sqrt(self.var + 1e-8)


class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = 0

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)

        # update count and moments
        n = x.shape[0]
        self.count += n
        delta = batch_mean - self.mean
        self.mean += delta * n / self.count
        m_a = self.var * (self.count - n)
        m_b = batch_var * n
        M2 = m_a + m_b + np.square(delta) * self.count * n / self.count
        self.var = M2 / self.count

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)
