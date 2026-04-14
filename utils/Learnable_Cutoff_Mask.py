import torch
import torch.nn as nn

class LearnableCutoffMask(nn.Module):
    def __init__(self, H, W, init_cutoff_ratio=0.3):
        super().__init__()
        self.H = H
        self.W = W
        self.cutoff_ratio = nn.Parameter(torch.tensor(init_cutoff_ratio))

    def forward(self, device='cuda'):
        yy, xx = torch.meshgrid(
            torch.arange(self.H, dtype=torch.float32, device=device) - self.H // 2,
            torch.arange(self.W, dtype=torch.float32, device=device) - self.W // 2,
            indexing='ij'
        )
        freq_radius = torch.sqrt(xx ** 2 + yy ** 2)
        max_radius = freq_radius.max()
        cutoff = torch.clamp(self.cutoff_ratio, 0.05, 0.95) * max_radius

        low_mask = (freq_radius <= cutoff).float()
        high_mask = 1.0 - low_mask

        return low_mask[None, None, :, :], high_mask[None, None, :, :]
