import torch
import torch.nn.functional as F
import random

class DegradationScheduler:
    def __init__(self, methods=None, severity=1):
        self.methods = methods or ['defocus', 'motion', 'zoom', 'fog', 'brightness', 'contrast']
        self.severity = severity

    def apply_random_degradation(self, x):
        method = random.choice(self.methods)
        return self._apply(x, method)

    def _apply(self, x, method):
        if method == 'defocus':
            return self.defocus_blur(x)
        elif method == 'motion':
            return self.motion_blur(x)
        elif method == 'zoom':
            return self.zoom_blur(x)
        elif method == 'fog':
            return self.fog(x)
        elif method == 'brightness':
            return self.brightness(x)
        elif method == 'contrast':
            return self.contrast(x)
        else:
            raise ValueError(f"Unknown degradation method: {method}")

    def defocus_blur(self, x):
        return F.gaussian_blur(x, kernel_size=[5, 5], sigma=[1.5, 1.5])

    def motion_blur(self, x, kernel_size=None):
        if kernel_size is None:
            kernel_size = [10, 3, 5, 8, 12, 15][self.severity - 1]
        B, C, H, W = x.shape
        weight = torch.zeros((1, 1, kernel_size, kernel_size), device=x.device)
        weight[:, :, kernel_size // 2, :] = 1.0 / kernel_size
        weight = weight.expand(C, 1, kernel_size, kernel_size)
        return F.conv2d(x, weight, padding=kernel_size // 2, groups=C)

    def zoom_blur(self, x):
        zoom_levels = [
            [1.0, 1.01, 1.02, 1.03, 1.04],
            [1.0, 1.02, 1.04, 1.06, 1.08],
            [1.0, 1.05, 1.1, 1.15],
            [1.0, 1.1, 1.2, 1.3],
            [1.0, 1.15, 1.3, 1.45],
        ][self.severity - 1]
        B, C, H, W = x.shape
        blurred = x.clone()
        for z in zoom_levels:
            size = [int(H * z), int(W * z)]
            zoomed = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
            zoomed = F.interpolate(zoomed, size=(H, W), mode='bilinear', align_corners=False)
            blurred += zoomed
        return blurred / (len(zoom_levels) + 1)

    def fog(self, x):
        fog_strengths = [1.5, 2.0, 2.5, 2.5, 3.0]
        decay_rates = [2, 2, 1.7, 1.5, 1.4]
        fog_strength = fog_strengths[self.severity - 1]
        decay = decay_rates[self.severity - 1]
        fog = torch.randn_like(x) * fog_strength
        fog = F.avg_pool2d(fog, kernel_size=31, stride=1, padding=15)
        return torch.clamp(x + fog, 0, 1)

    def brightness(self, x):
        delta = [.1, .2, .3, .4, .5][self.severity - 1]
        return torch.clamp(x + delta, 0, 1)

    def contrast(self, x):
        factor = [1.6, 2.2, 2.8, 3.4, 4.0][self.severity - 1]
        mean = x.mean(dim=(2, 3), keepdim=True)
        return torch.clamp((x - mean) * factor + mean, 0, 1)
