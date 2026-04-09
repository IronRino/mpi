"""Нейросетевые модели для MPI реконструкции"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MPIReconstructionCNN(nn.Module):
    """CNN для MPI реконструкции (архитектура UNet)"""

    def __init__(self, input_channels=4, output_channels=1, base_filters=32):
        super().__init__()

        # Энкодер
        self.enc1_conv = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        self.enc1_pool = nn.MaxPool2d(2)

        self.enc2_conv = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True)
        )
        self.enc2_pool = nn.MaxPool2d(2)

        self.enc3_conv = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True)
        )
        self.enc3_pool = nn.MaxPool2d(2)

        # Мост
        self.bridge = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 8, base_filters * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 8),
            nn.ReLU(inplace=True)
        )

        # Декодер
        self.dec3_up = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(base_filters * 4 + base_filters * 4, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True)
        )

        self.dec2_up = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(base_filters * 2 + base_filters * 2, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True)
        )

        self.dec1_up = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(base_filters + base_filters, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )

        # Финальные слои
        self.final_upsample = nn.Upsample(size=(51, 51), mode='bilinear', align_corners=False)
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_filters, base_filters // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters // 2, output_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Энкодер
        enc1 = self.enc1_conv(x)
        enc1_pooled = self.enc1_pool(enc1)

        enc2 = self.enc2_conv(enc1_pooled)
        enc2_pooled = self.enc2_pool(enc2)

        enc3 = self.enc3_conv(enc2_pooled)
        enc3_pooled = self.enc3_pool(enc3)

        # Мост
        bridge = self.bridge(enc3_pooled)

        # Декодер
        dec3_up = self.dec3_up(bridge)
        enc3_resized = F.interpolate(enc3, size=dec3_up.shape[2:], mode='bilinear', align_corners=False)
        dec3_concat = torch.cat([dec3_up, enc3_resized], dim=1)
        dec3 = self.dec3_conv(dec3_concat)

        dec2_up = self.dec2_up(dec3)
        enc2_resized = F.interpolate(enc2, size=dec2_up.shape[2:], mode='bilinear', align_corners=False)
        dec2_concat = torch.cat([dec2_up, enc2_resized], dim=1)
        dec2 = self.dec2_conv(dec2_concat)

        dec1_up = self.dec1_up(dec2)
        enc1_resized = F.interpolate(enc1, size=dec1_up.shape[2:], mode='bilinear', align_corners=False)
        dec1_concat = torch.cat([dec1_up, enc1_resized], dim=1)
        dec1 = self.dec1_conv(dec1_concat)

        # Финальные слои
        dec1_upsampled = self.final_upsample(dec1)
        output = self.final_conv(dec1_upsampled)

        return output


class TikhonovReconstructor:
    """Класс-обертка для метода Тихонова"""

    def __init__(self, system_matrix):
        self.SM = system_matrix

    def reconstruct(self, measurement, mu=0.001, kmax=100):
        """Реконструкция методом Тихонова"""
        import numpy as np
        from ReconstructionMethod import ReconstructionMethod as RM

        Meas = np.concatenate([measurement[0, :], measurement[1, :]])
        recImage = RM.Tikhonov_underdetermined(self.SM, Meas, mu, kmax)
        recImage = recImage.real.reshape((51, 51))

        if recImage.max() > 0:
            recImage = recImage / recImage.max()

        return recImage