"""Нейросетевые модели для MPI реконструкции"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MPIReconstructionCNN(nn.Module):
    """Стандартная CNN для MPI реконструкции (архитектура UNet)"""

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


class MoDLNetwork(nn.Module):
    """
    Полная сеть MoDL для MPI реконструкции
    Встраивает знание физической модели (системной матрицы) в архитектуру

    ВНИМАНИЕ: Для MPI системная матрица A имеет размер (M, N), где:
    - M = количество измерений (гармоник * каналы)
    - N = количество вокселей (nx * ny)

    Уравнение: A * x = y, где y - измерения (M), x - изображение (N)
    """
    def __init__(self, system_matrix, image_shape, n_iterations=5, lambda_param=0.01):
        super().__init__()

        # Системная матрица A размера (M, N)
        # Преобразуем комплексную матрицу в вещественную, разделяя real и imag
        if np.iscomplexobj(system_matrix):
            # Разделяем на реальную и мнимую части
            SM_real = np.real(system_matrix)
            SM_imag = np.imag(system_matrix)
            # Создаем расширенную матрицу [real; imag] размера (2M, N)
            self.system_matrix_extended = np.vstack([SM_real, SM_imag])
        else:
            self.system_matrix_extended = system_matrix

        self.M, self.N = self.system_matrix_extended.shape
        self.image_shape = image_shape
        self.n_iterations = n_iterations
        self.lambda_param = lambda_param

        print(f"MoDL: M={self.M}, N={self.N}, image_shape={image_shape}")

        # Конвертируем в тензор
        self.register_buffer('A', torch.tensor(self.system_matrix_extended, dtype=torch.float32))
        self.register_buffer('A_T', self.A.T)

        # Входной проекционный слой (измерения -> изображение)
        # Используем псевдообращение для начальной инициализации
        # Но можно и просто линейный слой
        self.input_projection = nn.Linear(self.M, self.N)

        # Инициализируем веса проекции как A^T (сопряженный оператор)
        with torch.no_grad():
            self.input_projection.weight.data = self.A_T.clone()
            self.input_projection.bias.data.zero_()

        # Денойзер на основе UNet
        self.denoiser = MoDLDenoiser(input_channels=1, base_filters=64)

        # Инициализационный слой (грубая реконструкция)
        self.initial_recon = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward_operator(self, x):
        """
        Прямой оператор A: изображение -> измерения
        x: (B, 1, H, W) -> выход: (B, M)
        """
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)  # (B, N)
        y_flat = x_flat @ self.A_T  # (B, M)
        return y_flat

    def adjoint_operator(self, y):
        """
        Сопряженный оператор A^T: измерения -> изображение
        y: (B, M) -> выход: (B, 1, H, W)
        """
        batch_size = y.shape[0]
        x_flat = y @ self.A  # (B, N)
        x = x_flat.view(batch_size, 1, self.image_shape[0], self.image_shape[1])
        return x

    def forward(self, measurements):
        """
        measurements: (B, M) - вектор измерений (вещественный, real+imag объединены)
        """
        batch_size = measurements.shape[0]

        # Способ 1: Проекция через обученный слой
        x_flat = self.input_projection(measurements)  # (B, N)
        x = x_flat.view(batch_size, 1, self.image_shape[0], self.image_shape[1])

        # Способ 2: Использование сопряженного оператора (можно раскомментировать)
        # x = self.adjoint_operator(measurements)

        # Улучшение через initial_recon
        x = self.initial_recon(x)

        # Итерации MoDL
        for _ in range(self.n_iterations):
            # Градиентный шаг: x - λ * A^T(A x - y)
            Ax = self.forward_operator(x)  # (B, M)
            residual = Ax - measurements   # (B, M)
            AT_residual = self.adjoint_operator(residual)  # (B, 1, H, W)
            grad_step = x - self.lambda_param * AT_residual

            # Денойзинг
            x = self.denoiser(grad_step)

        return x


class MoDLDenoiser(nn.Module):
    """Денойзер для MoDL сети"""
    def __init__(self, input_channels=1, base_filters=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.middle = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 2, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_filters, base_filters, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.bottleneck(x2)
        x4 = self.decoder(x3)

        # Обрезаем до исходного размера
        if x4.shape[-1] != x.shape[-1]:
            x4 = F.interpolate(x4, size=x.shape[-2:], mode='bilinear', align_corners=False)

        return x4


class DiffusionUNet(nn.Module):
    """UNet с временным embedding для диффузионной модели"""
    def __init__(self, in_channels=2, out_channels=1, base_filters=64):
        super().__init__()
        self.in_channels = in_channels

        # Временной embedding
        self.time_embed = nn.Sequential(
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, base_filters)
        )

        # Энкодер
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_filters),
            nn.SiLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_filters * 2),
            nn.SiLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_filters * 4),
            nn.SiLU()
        )

        # Декодер
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_filters * 2),
            nn.SiLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_filters),
            nn.SiLU()
        )
        self.out_conv = nn.Conv2d(base_filters, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        # Упрощенный временной embedding (для совместимости)
        batch_size = x.shape[0]
        t_emb = torch.zeros(batch_size, 256, device=x.device)
        if t is not None and len(t) == batch_size:
            # Простая синусоидальная позиционная кодировка
            for i in range(256):
                if i < t.shape[0]:
                    t_emb[:, i] = torch.sin(t.float() / (10000 ** (i / 256)))

        t_emb = self.time_embed(t_emb)

        # Энкодер
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        # Декодер
        x4 = self.dec1(x3)
        x5 = self.dec2(x4 + x2)

        output = self.out_conv(x5 + x1)

        return output


class DiffusionModel(nn.Module):
    """
    Генеративная диффузионная модель для MPI реконструкции
    """
    def __init__(self, denoiser, n_steps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.denoiser = denoiser
        self.n_steps = n_steps

        # Параметры диффузионного процесса
        self.beta = torch.linspace(beta_start, beta_end, n_steps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def q_sample(self, x0, t, noise=None):
        """Прямой процесс: добавление шума к изображению"""
        if noise is None:
            noise = torch.randn_like(x0)

        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

    def p_sample(self, x_t, t, condition=None):
        """Обратный процесс: удаление шума с учетом условий"""
        if condition is not None:
            x_input = torch.cat([x_t, condition], dim=1)
        else:
            x_input = x_t

        predicted_noise = self.denoiser(x_input, t)

        alpha_t = self.alpha[t].view(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        beta_t = self.beta[t].view(-1, 1, 1, 1)

        mean = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)

        if t[0] > 0:
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(beta_t) * noise
        else:
            return mean

    def forward(self, x0, condition=None):
        """Обучение: предсказание шума"""
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device)
        noise = torch.randn_like(x0)

        x_t = self.q_sample(x0, t, noise)

        if condition is not None:
            x_input = torch.cat([x_t, condition], dim=1)
        else:
            x_input = x_t

        predicted_noise = self.denoiser(x_input, t)

        return predicted_noise, noise


class CombinedHybridModel(nn.Module):
    """
    Гибридная модель: сочетает MoDL и Diffusion подходы
    """
    def __init__(self, system_matrix, image_shape, modl_iterations=3, diffusion_steps=50):
        super().__init__()
        self.modl = MoDLNetwork(system_matrix, image_shape, n_iterations=modl_iterations)
        self.diffusion_denoiser = DiffusionUNet(in_channels=2, out_channels=1)
        self.diffusion = DiffusionModel(self.diffusion_denoiser, n_steps=diffusion_steps)
        self.image_shape = image_shape

    def forward(self, measurements, return_intermediate=False):
        """
        measurements: (B, M) - вектор измерений
        """
        # Шаг 1: MoDL реконструкция
        modl_output = self.modl(measurements)

        # Шаг 2: Diffusion улучшение
        batch_size = measurements.shape[0]
        x_t = torch.randn(batch_size, 1, self.image_shape[0], self.image_shape[1]).to(measurements.device)

        for t in reversed(range(self.diffusion.n_steps)):
            t_tensor = torch.full((batch_size,), t, device=measurements.device, dtype=torch.long)
            condition = modl_output
            x_t = self.diffusion.p_sample(x_t, t_tensor, condition)

        if return_intermediate:
            return x_t, modl_output
        return x_t


class TikhonovReconstructor:
    """Класс-обертка для метода Тихонова"""

    def __init__(self, system_matrix):
        self.SM = system_matrix

    def reconstruct(self, measurement, mu=0.001, kmax=100):
        """Реконструкция методом Тихонова"""
        from ReconstructionMethod import ReconstructionMethod as RM

        Meas = np.concatenate([measurement[0, :], measurement[1, :]])
        recImage = RM.Tikhonov_underdetermined(self.SM, Meas, mu, kmax)
        recImage = recImage.real.reshape((51, 51))

        if recImage.max() > 0:
            recImage = recImage / recImage.max()

        return recImage