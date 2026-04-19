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


# ====================================================================
# МЕТОД 1: Chae (2017) - Однослойная полносвязная нейронная сеть
# Источник: Chae, B. G. "Neural network image reconstruction for
#           magnetic particle imaging." ETRI Journal, 2017.
# ====================================================================
class ChaeSingleLayerNN(nn.Module):
    """
    Однослойная полносвязная нейронная сеть для реконструкции MPI
    Chae (2017) - ETRI Journal

    Архитектура: полносвязный слой с нелинейностью (ReLU)
    Вход: вектор измерений (реальная + мнимая части)
    Выход: вектор изображения
    """
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()

        # По оригинальной статье используется один скрытый слой
        if hidden_dim is None:
            # По умолчанию: один слой (линейная регрессия)
            self.network = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(inplace=True)
            )
        else:
            # Два слоя (один скрытый)
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim),
                nn.Sigmoid()
            )

    def forward(self, x):
        return self.network(x)


# ====================================================================
# МЕТОД 2: Dittmer et al. (2020) - Deep Image Prior (DIP)
# Источник: Dittmer, S., et al. "Deep image prior for 3D magnetic
#           particle imaging." arXiv, 2020.
# ====================================================================
class DeepImagePrior(nn.Module):
    """
    Deep Image Prior (DIP) для MPI реконструкции
    Dittmer et al. (2020) - arXiv:2007.01593

    Идея: Использовать структуру генеративной сети как регуляризатор
    Сеть обучается на одном изображении для соответствия измерениям
    """
    def __init__(self, image_shape, latent_dim=100, n_channels=64):
        super().__init__()
        self.image_shape = image_shape
        self.latent_dim = latent_dim

        # Генеративная сеть (архитектура как в оригинале)
        self.network = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, n_channels * 8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(n_channels * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(n_channels * 8, n_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n_channels * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(n_channels * 4, n_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n_channels * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(n_channels * 2, n_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(n_channels, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        # Адаптивный выходной размер
        self.output_conv = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, z):
        # z: латентный вектор (batch, latent_dim, 1, 1)
        x = self.network(z)
        # Изменение размера до нужного
        if x.shape[-2:] != self.image_shape:
            x = F.interpolate(x, size=self.image_shape, mode='bilinear', align_corners=False)
        return x

    def generate_random_latent(self, batch_size=1):
        """Генерация случайного латентного вектора"""
        return torch.randn(batch_size, self.latent_dim, 1, 1)


# ====================================================================
# МЕТОД 3: Shang et al. (2020) - CNN для MPI реконструкции
# Источник: Shang, Y., et al. "Deep learning for improving the
#           spatial resolution of magnetic particle imaging."
#           Physics in Medicine & Biology, 2020.
# ====================================================================
class ShangCNN(nn.Module):
    """
    CNN модель для улучшения пространственного разрешения MPI
    Shang et al. (2020) - Physics in Medicine & Biology

    Архитектура: 5-слойная CNN с residual connections
    """
    def __init__(self, input_channels=1, output_channels=1, base_filters=64):
        super().__init__()

        # Сверточные слои
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Conv2d(base_filters, output_channels, kernel_size=3, padding=1)

        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        # Сохраняем вход для residual connection
        residual = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Residual connection (как в оригинальной статье)
        x = x + residual

        return self.output_activation(x)


# ====================================================================
# МЕТОД 4: Wu et al. (2023) - PGNet (Projection Generation Network)
# Источник: Wu, X., et al. "PGNet: Projection generative network for
#           sparse-view reconstruction of projection-based magnetic
#           particle imaging." Medical Physics, 2023.
# ====================================================================
class PGNet(nn.Module):
    """
    Projection Generation Network для реконструкции при разреженных проекциях
    Wu et al. (2023) - Medical Physics

    Архитектура: Генеративная сеть с attention механизмом
    """
    def __init__(self, input_dim, output_shape, hidden_dim=512):
        super().__init__()
        self.output_shape = output_shape
        output_dim = output_shape[0] * output_shape[1]

        # Энкодер проекций
        self.projection_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Генератор изображений
        self.image_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

        # Восстановление пространственной структуры
        self.spatial_reshape = lambda x: x.view(-1, 1, output_shape[0], output_shape[1])

    def forward(self, measurements):
        # measurements: (batch, input_dim)
        encoded = self.projection_encoder(measurements)
        generated = self.image_generator(encoded)
        return self.spatial_reshape(generated)


# ====================================================================
# МЕТОД 5: Güngör et al. (2024) - DEQ-MPI
# Источник: Güngör, A., et al. "DEQ-MPI: A deep equilibrium
#           reconstruction with learned consistency for magnetic
#           particle imaging." IEEE TMI, 2024.
# ====================================================================
class DEQMPI(nn.Module):
    """
    Deep Equilibrium Model для MPI реконструкции
    Güngör et al. (2024) - IEEE Transactions on Medical Imaging

    Идея: Использовать неявные слои для моделирования равновесного состояния
    """
    def __init__(self, system_matrix, image_shape, n_iterations=10, lambda_param=0.1):
        super().__init__()

        # Подготовка системной матрицы
        if np.iscomplexobj(system_matrix):
            SM_real = np.real(system_matrix)
            SM_imag = np.imag(system_matrix)
            self.system_matrix_extended = np.vstack([SM_real, SM_imag])
        else:
            self.system_matrix_extended = system_matrix

        self.M, self.N = self.system_matrix_extended.shape
        self.image_shape = image_shape
        self.n_iterations = n_iterations
        self.lambda_param = lambda_param

        # Регистрация матриц
        self.register_buffer('A', torch.tensor(self.system_matrix_extended, dtype=torch.float32))
        self.register_buffer('A_T', self.A.T)

        # Обучаемый оператор согласованности данных
        self.data_consistency = nn.Sequential(
            nn.Linear(self.M, self.M),
            nn.ReLU(inplace=True),
            nn.Linear(self.M, self.M)
        )

        # Обучаемый денойзер
        self.denoiser = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Начальная реконструкция через псевдообращение
        self.initial_recon = nn.Linear(self.M, self.N)
        with torch.no_grad():
            self.initial_recon.weight.data = self.A_T.clone()

    def forward_operator(self, x):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        return x_flat @ self.A_T

    def adjoint_operator(self, y):
        batch_size = y.shape[0]
        x_flat = y @ self.A
        return x_flat.view(batch_size, 1, self.image_shape[0], self.image_shape[1])

    def forward(self, measurements):
        # Начальная реконструкция
        x_flat = self.initial_recon(measurements)
        x = x_flat.view(-1, 1, self.image_shape[0], self.image_shape[1])

        # Итерации DEQ (фиксированная точка)
        for _ in range(self.n_iterations):
            # Шаг согласованности данных с обученным оператором
            Ax = self.forward_operator(x)
            residual = self.data_consistency(Ax - measurements)
            AT_residual = self.adjoint_operator(residual)

            # Обновление с денойзингом
            x = x - self.lambda_param * AT_residual
            x = self.denoiser(x)

        return x


# ====================================================================
# МЕТОД 6: KatsMarc (Алгоритм Кацмарца - ART)
# Источник: Kaczmarz, S. (1937). "Angenäherte Auflösung von
#           Systemen linearer Gleichungen." Bulletin International
#           de l'Académie Polonaise des Sciences et des Lettres.
# ====================================================================
class KatsMarcAlgorithm:
    """
    Алгоритм Кацмарца (Algebraic Reconstruction Technique - ART)
    Kaczmarz (1937) - Метод последовательных проекций

    Для системы A x = b, алгоритм итеративно проектирует решение
    на гиперплоскости, заданные строками матрицы
    """
    def __init__(self, system_matrix):
        self.A = system_matrix
        self.M, self.N = system_matrix.shape

    def reconstruct(self, b, n_iterations=10, relaxation=1.0):
        """
        Реконструкция алгоритмом Кацмарца

        Args:
            b: вектор измерений (M,)
            n_iterations: количество полных проходов по строкам
            relaxation: параметр релаксации (0 < relaxation <= 2)
        """
        x = np.zeros(self.N)

        for iteration in range(n_iterations):
            for i in range(self.M):
                a_i = self.A[i, :]
                if np.linalg.norm(a_i) > 1e-10:
                    # Проекция на гиперплоскость a_i * x = b[i]
                    residual = b[i] - np.dot(a_i, x)
                    x = x + relaxation * (residual / np.linalg.norm(a_i)**2) * a_i

        return x


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

    def __init__(self, in_channels=2, out_channels=1, base_filters=64, image_size=51):
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size

        # Временной embedding
        self.time_embed = nn.Sequential(
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, base_filters)
        )

        # Энкодер - используем адаптивные размеры
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_filters),
            nn.SiLU(),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_filters),
            nn.SiLU()
        )

        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_filters * 2),
            nn.SiLU()
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_filters * 2),
            nn.SiLU(),
            nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_filters * 2),
            nn.SiLU()
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_filters * 4),
            nn.SiLU()
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_filters * 4),
            nn.SiLU(),
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_filters * 4),
            nn.SiLU()
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_filters * 8),
            nn.SiLU(),
            nn.Conv2d(base_filters * 8, base_filters * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_filters * 4),
            nn.SiLU()
        )

        # Декодер
        self.up1 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_filters * 2),
            nn.SiLU(),
            nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_filters * 2),
            nn.SiLU()
        )

        self.up2 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_filters),
            nn.SiLU(),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_filters),
            nn.SiLU()
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(base_filters, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Финальный апсемплинг до нужного размера
        self.final_upsample = nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=False)

    def forward(self, x, t):
        """
        Args:
            x: входной тензор формы (B, C, H, W), где C = in_channels (2)
            t: временной шаг (B,)
        """
        batch_size = x.shape[0]
        original_size = x.shape[-2:]

        # Временной embedding
        t_emb = torch.zeros(batch_size, 256, device=x.device)
        if t is not None and len(t) == batch_size:
            t_float = t.float().unsqueeze(1)
            div_term = torch.exp(torch.arange(0, 256, 2, device=x.device).float() * (-np.log(10000.0) / 256))
            t_emb[:, 0::2] = torch.sin(t_float * div_term)
            t_emb[:, 1::2] = torch.cos(t_float * div_term)

        t_emb = self.time_embed(t_emb)
        # Добавляем временной embedding к признакам через адаптивный слой
        # (упрощенно - пропускаем, так как требует дополнительной архитектуры)

        # Энкодер
        x1 = self.enc1(x)  # (B, base_filters, H, W)

        x2 = self.down1(x1)  # (B, base_filters*2, H/2, W/2)
        x3 = self.enc2(x2)  # (B, base_filters*2, H/2, W/2)

        x4 = self.down2(x3)  # (B, base_filters*4, H/4, W/4)
        x5 = self.enc3(x4)  # (B, base_filters*4, H/4, W/4)

        # Bottleneck
        x6 = self.bottleneck(x5)  # (B, base_filters*4, H/4, W/4)

        # Декодер
        x7 = self.up1(x6)  # (B, base_filters*2, H/2, W/2)
        # Обрезаем или интерполируем для совпадения размеров
        if x7.shape[-2:] != x3.shape[-2:]:
            x7 = F.interpolate(x7, size=x3.shape[-2:], mode='bilinear', align_corners=False)
        x7 = torch.cat([x7, x3], dim=1)  # (B, base_filters*4, H/2, W/2)
        x7 = self.dec1(x7)  # (B, base_filters*2, H/2, W/2)

        x8 = self.up2(x7)  # (B, base_filters, H, W)
        if x8.shape[-2:] != x1.shape[-2:]:
            x8 = F.interpolate(x8, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        x8 = torch.cat([x8, x1], dim=1)  # (B, base_filters*2, H, W)
        x8 = self.dec2(x8)  # (B, base_filters, H, W)

        output = self.out_conv(x8)  # (B, 1, H, W)

        # Финальный апсемплинг до целевого размера
        if output.shape[-2:] != original_size:
            output = self.final_upsample(output)

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
        self.register_buffer('beta', torch.linspace(beta_start, beta_end, n_steps))
        self.register_buffer('alpha', 1.0 - self.beta)
        self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim=0))

    def q_sample(self, x0, t, noise=None):
        """Прямой процесс: добавление шума к изображению"""
        if noise is None:
            noise = torch.randn_like(x0)

        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

    def p_sample(self, x_t, t, condition=None):
        """Обратный процесс: удаление шума с учетом условий"""
        # Приводим condition к размеру x_t если необходимо
        if condition is not None:
            if condition.shape[-2:] != x_t.shape[-2:]:
                condition = F.interpolate(condition, size=x_t.shape[-2:], mode='bilinear', align_corners=False)
            # Объединяем по каналам: x_t (1 канал) + condition (1 канал) = 2 канала
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
        """Обучение: предсказание шума

        Args:
            x0: чистое изображение (target) - форма (B, 1, H, W)
            condition: условие (измерения) - форма (B, C, H, W), где C может быть 4
        """
        batch_size = x0.shape[0]
        device = x0.device

        t = torch.randint(0, self.n_steps, (batch_size,), device=device)
        noise = torch.randn_like(x0)

        x_t = self.q_sample(x0, t, noise)

        # Для диффузионной модели условие должно иметь 1 канал
        # Усредняем каналы условия, если их больше 1
        if condition is not None:
            if condition.shape[-2:] != x_t.shape[-2:]:
                condition = F.interpolate(condition, size=x_t.shape[-2:], mode='bilinear', align_corners=False)

            # Если у условия больше 1 канала, усредняем их
            if condition.shape[1] > 1:
                condition = condition.mean(dim=1, keepdim=True)

            x_input = torch.cat([x_t, condition], dim=1)  # Теперь (B, 2, H, W)
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
        self.diffusion_denoiser = DiffusionUNet(in_channels=2, out_channels=1, image_size=51)
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