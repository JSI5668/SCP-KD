import torch
import torch.nn as nn
import torchvision.models as models


class FeatureExtractor(nn.Module):
    def __init__(self, backbone_type="resnet18"):
        super(FeatureExtractor, self).__init__()

        if backbone_type == "resnet18":
            self.backbone = models.resnet18(pretrained=True)
            self.backbone.fc = nn.Identity()  # Fully Connected Layer 제거
        elif backbone_type == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=True)
            self.backbone.classifier = nn.Identity()
        elif backbone_type == "mobilenet_v3":
            self.backbone = models.mobilenet_v3_small(pretrained=True)
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError("Unsupported backbone type")

    def forward(self, x):
        features = self.backbone(x)

        # (B, C) 형태라면 (B, C, 1, 1)로 변경
        if len(features.shape) == 2:
            features = features.unsqueeze(-1).unsqueeze(-1)

        return features



class OutpaintingHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutpaintingHead, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # (batch, channels) 형태의 벡터를 (batch, channels, height, width) 형태로 변환
        if len(x.shape) == 2:  # (B, C) -> (B, C, 1, 1)
            x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.deconv3(x)
        return x


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SegmentationHead, self).__init__()
        self.conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, num_classes, kernel_size=1)

        self.deconv1 = nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))

        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return x

# class PolarAwareCrossAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(PolarAwareCrossAttention, self).__init__()
#         self.query = nn.Linear(in_channels, in_channels)
#         self.key = nn.Linear(192, 192)
#         self.value = nn.Linear(192, 192)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x1, x2):
#         # (B, C, H, W) → (B, C)
#         x1 = x1.view(x1.shape[0], -1)
#         x2 = x2.view(x2.shape[0], -1)
#
#         q = self.query(x1)  # (B, C) → (B, C)
#         k = self.key(x2)  # (B, C) → (B, C)
#         v = self.value(x2)  # (B, C) → (B, C)
#
#         attention = self.softmax(torch.matmul(q, k.transpose(-2, -1)))  # (B, C) @ (B, C).T
#         output = torch.matmul(attention, v)
#         return output


class FishDreamer(nn.Module):
    def __init__(self, num_classes=12):
        super(FishDreamer, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.segmentation_head = SegmentationHead(512, num_classes)
        self.outpainting_head = OutpaintingHead(512, 3)
        # self.pca = PolarAwareCrossAttention(12)
        # Upsampling Layer 추가
        # self.upsample = nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False)
        self.upsample = nn.Upsample(size=(28, 28), mode="bilinear", align_corners=False)

    def forward(self, x):
        features = self.feature_extractor(x)

        # 업샘플링 적용
        features = self.upsample(features)

        outpainting = self.outpainting_head(features)
        segmentation = self.segmentation_head(features)

        return segmentation, outpainting

