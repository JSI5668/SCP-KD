import torch

class CustomFeatureScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule_type='linear', device='cpu'):
        self.num_train_timesteps = num_train_timesteps
        self.device = device  # ⭐ 추가: 디바이스 정보 저장

        if schedule_type == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, device=device)
        elif schedule_type == 'cosine':
            timesteps = torch.arange(num_train_timesteps + 1, dtype=torch.float64, device=device) / num_train_timesteps
            alphas_cumprod = torch.cos((timesteps + 0.008) / 1.008 * torch.pi / 2) ** 2
            self.betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = self.betas.to(device)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alphas_cumprod).to(device)

    def add_noise(self, clean_feature, noise, t):
        """
        clean_feature: (B, C, H, W)
        noise: same shape
        t: (B,) tensor of timestep
        """
        # 필요한 계수를 batch에 맞게 indexing
        sqrt_alpha_hat = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha_hat * clean_feature + sqrt_one_minus_alpha_hat * noise
