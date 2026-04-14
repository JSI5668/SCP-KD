import torch
import torch.nn as nn
import math

class TimestepEmbedding(nn.Module):
    def __init__(self, num_channels, embedding_dim=None):
        super().__init__()
        self.embedding_dim = embedding_dim or num_channels

        self.linear = nn.Sequential(
            nn.Linear(self.embedding_dim, num_channels),
            nn.ReLU(),
            nn.Linear(num_channels, num_channels),
        )

    def forward(self, t):
        """
        t: (B,) int tensor of timesteps
        return: (B, num_channels)
        """
        half_dim = self.embedding_dim // 2
        device = t.device
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]  # shape: (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # shape: (B, embedding_dim)

        return self.linear(emb)  # project to (B, num_channels)

class DenoiseBlock(nn.Module):
    def __init__(self, in_channels, embed_dim=128):
        super().__init__()
        self.t_embed = TimestepEmbedding(num_channels=embed_dim)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels + embed_dim, in_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )

    def forward(self, x, t):
        t_emb = self.t_embed(t)  # (B, embed_dim)
        t_emb = t_emb[..., None, None].expand(-1, -1, x.shape[2], x.shape[3])
        x_cat = torch.cat([x, t_emb], dim=1)
        return self.block(x_cat)


class DenoiseBlockLF(nn.Module):
    def __init__(self, in_channels, embed_dim=128):
        super().__init__()
        self.t_embed = TimestepEmbedding(num_channels=embed_dim)

        # Larger receptive field for global restoration
        self.block = nn.Sequential(
            nn.Conv2d(in_channels + embed_dim, in_channels, kernel_size=5, padding=2),  # 더 넓은 kernel
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2),
        )

    def forward(self, x, t):
        t_emb = self.t_embed(t)
        t_emb = t_emb[..., None, None].expand(-1, -1, x.shape[2], x.shape[3])
        x_cat = torch.cat([x, t_emb], dim=1)
        return self.block(x_cat)
