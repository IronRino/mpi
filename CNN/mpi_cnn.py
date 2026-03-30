import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split
import time

# Создаем директории
os.makedirs('./DATA/dataset', exist_ok=True)
os.makedirs('./DATA/models', exist_ok=True)
os.makedirs('./DATA/results', exist_ok=True)


class MPIDatasetGenerator:
    """Генератор датасета для обучения CNN на MPI реконструкции"""

    def __init__(self, system_matrix_path='./../DATA/SystemMatrix.h5'):
        self.load_system_matrix(system_matrix_path)

    def load_system_matrix(self, path):
        """Загрузка системной матрицы"""
        print("Загрузка системной матрицы...")
        fSM = h5py.File(path, 'r')
        S_data_r = fSM['/measurement/data/r'][:]
        S_data_i = fSM['/measurement/data/i'][:]
        S = S_data_r + 1j * S_data_i
        isBG = fSM['/measurement/isBackgroundFrame'][:].squeeze()
        S = S[:, :, isBG == 0]
        self.SM = S.reshape(S.shape[0] * S.shape[1], S.shape[2])

        number_Position = fSM['/calibration/size'][:].squeeze()
        self.nx, self.ny = int(number_Position[0]), int(number_Position[1])
        self.image_shape = (self.nx, self.ny)

        print(f"Размер SM: {self.SM.shape}")
        print(f"Размер изображения: {self.image_shape}")
        fSM.close()

    def generate_two_droplets(self, radius, distance, intensity1=0.7, intensity2=0.7):
        """Генерация изображения с двумя каплями"""
        image = np.zeros((self.nx, self.ny))

        # Координаты сетки в диапазоне [-1, 1]
        x = np.linspace(-1, 1, self.nx)
        y = np.linspace(-1, 1, self.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Центры капель
        center1_x = -distance / 2
        center2_x = distance / 2

        # Создаем капли с плавными краями (гауссово распределение)
        sigma = radius / 2.5  # Для плавного спада

        # Первая капля
        r1 = np.sqrt((X - center1_x) ** 2 + Y ** 2)
        image += intensity1 * np.exp(-(r1 ** 2) / (2 * sigma ** 2))

        # Вторая капля
        r2 = np.sqrt((X - center2_x) ** 2 + Y ** 2)
        image += intensity2 * np.exp(-(r2 ** 2) / (2 * sigma ** 2))

        # Нормализация
        if image.max() > 0:
            image = image / image.max()

        return image

    def generate_random_image(self):
        """Генерация случайного изображения для обучения"""
        # Случайное количество капель (1-4)
        num_droplets = np.random.randint(1, 5)

        image = np.zeros((self.nx, self.ny))
        x = np.linspace(-1, 1, self.nx)
        y = np.linspace(-1, 1, self.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        for _ in range(num_droplets):
            # Случайные параметры
            center_x = np.random.uniform(-0.8, 0.8)
            center_y = np.random.uniform(-0.8, 0.8)
            radius = np.random.uniform(0.05, 0.3)
            intensity = np.random.uniform(0.3, 1.0)
            sigma = radius / 2.5

            # Гауссова капля
            r = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
            droplet = intensity * np.exp(-(r ** 2) / (2 * sigma ** 2))

            # Добавляем с проверкой перекрытия
            overlap_threshold = 0.3
            if np.max(image + droplet) <= 1.0 + overlap_threshold:
                image = np.maximum(image, droplet)
            else:
                # Уменьшаем интенсивность если перекрытие
                image = image + 0.5 * droplet

        # Нормализация
        if image.max() > 0:
            image = image / image.max()

        return image

    def generate_measurement(self, image):
        """Генерация измерений из изображения"""
        image_vector = image.reshape(-1, 1)
        measurement = self.SM @ image_vector
        measurement_reshaped = measurement.reshape(2, -1)
        return measurement_reshaped

    def create_dataset(self, num_samples=5000, test_size=0.2):
        """Создание полного датасета"""
        print(f"Создание датасета из {num_samples} образцов...")

        images = []
        measurements = []

        for i in range(num_samples):
            if i % 500 == 0:
                print(f"  Создано {i}/{num_samples} образцов...")

            # 70% - случайные изображения, 30% - две капли
            if np.random.random() < 0.7:
                image = self.generate_random_image()
            else:
                radius = np.random.uniform(0.05, 0.3)
                distance = np.random.uniform(radius, min(2 * radius, 1.5))
                image = self.generate_two_droplets(radius, distance)

            measurement = self.generate_measurement(image)

            images.append(image)
            measurements.append(measurement)

        images = np.array(images, dtype=np.float32)
        measurements = np.array(measurements, dtype=np.complex64)

        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            measurements, images, test_size=test_size, random_state=42
        )

        print(f"Размеры датасета:")
        print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")

        # Сохранение датасета
        np.save('./DATA/dataset/X_train.npy', X_train)
        np.save('./DATA/dataset/X_test.npy', X_test)
        np.save('./DATA/dataset/y_train.npy', y_train)
        np.save('./DATA/dataset/y_test.npy', y_test)

        # Сохранение системной матрицы в удобном формате
        np.save('./DATA/dataset/SM.npy', self.SM)
        np.save('./DATA/dataset/image_shape.npy', self.image_shape)

        return X_train, X_test, y_train, y_test


class MPIReconstructionCNN(nn.Module):
    """Исправленная CNN для MPI реконструкции"""

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
        self.enc1_pool = nn.MaxPool2d(2)  # 36x36 -> 18x18

        self.enc2_conv = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True)
        )
        self.enc2_pool = nn.MaxPool2d(2)  # 18x18 -> 9x9

        self.enc3_conv = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True)
        )
        self.enc3_pool = nn.MaxPool2d(2)  # 9x9 -> 4x4 (округление)

        # Мост
        self.bridge = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 8, base_filters * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 8),
            nn.ReLU(inplace=True)
        )

        # Декодер - уровень 3 (4x4 -> 8x8)
        self.dec3_up = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3_conv = nn.Sequential(
            # Вход: bridge_up(256->128) + enc3(64) = 192 каналов
            nn.Conv2d(base_filters * 4 + base_filters * 4, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True)
        )

        # Декодер - уровень 2 (8x8 -> 16x16)
        self.dec2_up = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2_conv = nn.Sequential(
            # Вход: dec3_up(64->32) + enc2(32) = 64 канала
            nn.Conv2d(base_filters * 2 + base_filters * 2, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True)
        )

        # Декодер - уровень 1 (16x16 -> 32x32)
        self.dec1_up = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1_conv = nn.Sequential(
            # Вход: dec2_up(32->16) + enc1(16) = 32 канала
            nn.Conv2d(base_filters + base_filters, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )

        # Финальные слои (32x32 -> 51x51)
        self.final_upsample = nn.Upsample(size=(51, 51), mode='bilinear', align_corners=False)
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_filters, base_filters // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters // 2, output_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Инициализация весов
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
        enc1 = self.enc1_conv(x)  # 4x36x36 -> 32x36x36
        enc1_pooled = self.enc1_pool(enc1)  # 32x36x36 -> 32x18x18

        enc2 = self.enc2_conv(enc1_pooled)  # 32x18x18 -> 64x18x18
        enc2_pooled = self.enc2_pool(enc2)  # 64x18x18 -> 64x9x9

        enc3 = self.enc3_conv(enc2_pooled)  # 64x9x9 -> 128x9x9
        enc3_pooled = self.enc3_pool(enc3)  # 128x9x9 -> 128x4x4

        # Мост
        bridge = self.bridge(enc3_pooled)  # 128x4x4 -> 256x4x4

        # Декодер уровень 3
        dec3_up = self.dec3_up(bridge)  # 256x4x4 -> 128x8x8
        # Изменяем размер enc3 до 8x8
        enc3_resized = F.interpolate(enc3, size=dec3_up.shape[2:], mode='bilinear', align_corners=False)
        dec3_concat = torch.cat([dec3_up, enc3_resized], dim=1)  # 128 + 128 = 256
        dec3 = self.dec3_conv(dec3_concat)  # 256 -> 128

        # Декодер уровень 2
        dec2_up = self.dec2_up(dec3)  # 128x8x8 -> 64x16x16
        # Изменяем размер enc2 до 16x16
        enc2_resized = F.interpolate(enc2, size=dec2_up.shape[2:], mode='bilinear', align_corners=False)
        dec2_concat = torch.cat([dec2_up, enc2_resized], dim=1)  # 64 + 64 = 128
        dec2 = self.dec2_conv(dec2_concat)  # 128 -> 64

        # Декодер уровень 1
        dec1_up = self.dec1_up(dec2)  # 64x16x16 -> 32x32x32
        # Изменяем размер enc1 до 32x32
        enc1_resized = F.interpolate(enc1, size=dec1_up.shape[2:], mode='bilinear', align_corners=False)
        dec1_concat = torch.cat([dec1_up, enc1_resized], dim=1)  # 32 + 32 = 64
        dec1 = self.dec1_conv(dec1_concat)  # 64 -> 32

        # Финальные слои
        dec1_upsampled = self.final_upsample(dec1)  # 32x32x32 -> 32x51x51
        output = self.final_conv(dec1_upsampled)  # 32x51x51 -> 1x51x51

        return output


class MPIDataset(Dataset):
    """Датасет для MPI реконструкции - ИСПРАВЛЕННЫЙ"""

    def __init__(self, X_path, y_path):
        self.X = np.load(X_path)  # Измерения: (samples, 2, n_measurements)
        self.y = np.load(y_path)  # Изображения: (samples, nx, ny)

        # Находим ближайший квадратный размер для измерений
        self.n_measurements = self.X.shape[2]
        self.measurement_size = self._find_square_size(self.n_measurements)

    def _find_square_size(self, n):
        """Находит ближайший квадратный размер"""
        size = int(np.sqrt(n))
        if size * size == n:
            return size
        else:
            return size + 1

    def _prepare_measurement(self, measurement):
        """Подготовка измерений для CNN"""
        # measurement: (2, n_measurements)
        real_part = measurement.real  # (2, n_measurements)
        imag_part = measurement.imag  # (2, n_measurements)

        # Объединяем реальную и мнимую части
        combined = np.concatenate([real_part, imag_part], axis=0)  # (4, n_measurements)

        # Преобразуем в квадратную форму
        target_size = self.measurement_size * self.measurement_size
        if self.n_measurements < target_size:
            # Дополняем нулями
            padded = np.zeros((4, target_size), dtype=np.float32)
            padded[:, :self.n_measurements] = combined
            combined = padded

        # Reshape в квадрат (4, size, size)
        measurement_square = combined.reshape(4, self.measurement_size, self.measurement_size)

        return measurement_square

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Измерения
        measurement = self.X[idx]  # (2, n_measurements)
        measurement_prepared = self._prepare_measurement(measurement)  # (4, size, size)

        # Целевое изображение
        target = self.y[idx]  # (nx, ny)

        # Добавляем размерность канала
        target = np.expand_dims(target, axis=0)  # (1, nx, ny)

        return (torch.tensor(measurement_prepared, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32))


class MPITrainer:
    """Тренировка CNN для MPI реконструкции - ИСПРАВЛЕННЫЙ"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

    def create_model(self, input_channels=4, output_channels=1):
        """Создание модели"""
        self.model = MPIReconstructionCNN(input_channels, output_channels)
        self.model.to(self.device)
        return self.model

    def setup_training(self, learning_rate=1e-3, weight_decay=1e-5):
        """Настройка обучения"""
        # Простая MSE loss
        self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate,
                                          weight_decay=weight_decay)

        # Убираем verbose параметр
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

    def train_epoch(self, train_loader):
        """Одна эпоха обучения"""
        self.model.train()
        total_loss = 0

        for measurements, targets in train_loader:
            measurements = measurements.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(measurements)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        """Валидация"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for measurements, targets in val_loader:
                measurements = measurements.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(measurements)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader, epochs=50, save_path='./DATA/models/mpi_cnn_best.pth'):
        """Полный цикл обучения"""
        print("Начало обучения CNN...")

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')

        for epoch in range(epochs):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            self.scheduler.step(val_loss)

            # Сохраняем лучшую модель
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, save_path)
                print(f"  Сохранена лучшая модель (epoch {epoch + 1}, val_loss: {val_loss:.6f})")

            epoch_time = time.time() - start_time

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, "
                      f"Time: {epoch_time:.1f}s")

        # Загрузка лучшей модели
        if os.path.exists(save_path):
            checkpoint = torch.load(save_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Загружена лучшая модель из эпохи {checkpoint['epoch'] + 1}")

        return train_losses, val_losses

    def predict(self, measurement):
        """Предсказание на новых данных"""
        self.model.eval()

        with torch.no_grad():
            # Подготовка измерений
            if isinstance(measurement, np.ndarray):
                # measurement: (2, n_measurements)
                real_part = measurement.real  # (2, n_measurements)
                imag_part = measurement.imag  # (2, n_measurements)
                combined = np.concatenate([real_part, imag_part], axis=0)  # (4, n_measurements)

                # Находим ближайший квадратный размер
                n_measurements = measurement.shape[1]
                size = int(np.sqrt(n_measurements))
                if size * size != n_measurements:
                    size += 1
                    target_size = size * size

                    # Дополняем нулями
                    padded = np.zeros((4, target_size), dtype=np.float32)
                    padded[:, :n_measurements] = combined
                    combined = padded

                # Reshape
                measurement_reshaped = combined.reshape(1, 4, size, size)
                measurement_tensor = torch.tensor(measurement_reshaped, dtype=torch.float32)
            else:
                measurement_tensor = measurement

            measurement_tensor = measurement_tensor.to(self.device)
            output = self.model(measurement_tensor)

        return output.cpu().numpy()


# Создание и обучение модели
def train_cnn_model():
    """Функция обучения CNN модели - ИСПРАВЛЕННАЯ"""

    # Загрузка датасета
    train_dataset = MPIDataset('./DATA/dataset/X_train.npy', './DATA/dataset/y_train.npy')
    test_dataset = MPIDataset('./DATA/dataset/X_test.npy', './DATA/dataset/y_test.npy')

    # Проверка размеров
    sample_measurement, sample_target = train_dataset[0]
    print(f"Размер входных данных: {sample_measurement.shape}")
    print(f"Размер выходных данных: {sample_target.shape}")

    # Создание DataLoader
    batch_size = 8  # Уменьшил batch size для экономии памяти
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Создание и обучение модели
    trainer = MPITrainer()
    model = trainer.create_model(input_channels=4)  # 4 канала: real1, imag1, real2, imag2
    trainer.setup_training(learning_rate=1e-3)

    train_losses, val_losses = trainer.train(
        train_loader, val_loader,
        epochs=30,  # Уменьшил количество эпох для быстрого теста
        save_path='./DATA/models/mpi_cnn_best.pth'
    )

    # График обучения
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('CNN Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('./DATA/results/cnn_training_curve.png', dpi=150, bbox_inches='tight')
    plt.show()

    return trainer


class MPIReconstructionComparator:
    """Сравнение методов реконструкции: Тихонов vs CNN"""

    def __init__(self, system_matrix_path='./../DATA/SystemMatrix.h5'):
        self.load_system_matrix(system_matrix_path)
        self.cnn_trainer = None
        self.metrics_history = []

    def load_system_matrix(self, path):
        """Загрузка системной матрицы"""
        fSM = h5py.File(path, 'r')
        S_data_r = fSM['/measurement/data/r'][:]
        S_data_i = fSM['/measurement/data/i'][:]
        S = S_data_r + 1j * S_data_i
        isBG = fSM['/measurement/isBackgroundFrame'][:].squeeze()
        S = S[:, :, isBG == 0]
        self.SM = S.reshape(S.shape[0] * S.shape[1], S.shape[2])

        number_Position = fSM['/calibration/size'][:].squeeze()
        self.nx, self.ny = int(number_Position[0]), int(number_Position[1])
        self.image_shape = (self.nx, self.ny)

        fSM.close()

        print(f"Размер изображения: {self.image_shape}")
        print(f"Размер системной матрицы: {self.SM.shape}")

    def set_cnn_model(self, cnn_trainer):
        """Установка обученной CNN модели"""
        self.cnn_trainer = cnn_trainer

    def generate_test_case(self, radius=0.2, distance=0.2, intensity1=0.7, intensity2=0.7):
        """Генерация тестового случая с двумя каплями"""
        image = np.zeros((self.nx, self.ny))

        # Координаты сетки в диапазоне [-1, 1]
        x = np.linspace(-1, 1, self.nx)
        y = np.linspace(-1, 1, self.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Центры капель
        center1_x = -distance / 2
        center2_x = distance / 2

        # Создаем капли с плавными краями
        sigma = radius / 2.5

        # Первая капля
        r1 = np.sqrt((X - center1_x) ** 2 + Y ** 2)
        image += intensity1 * np.exp(-(r1 ** 2) / (2 * sigma ** 2))

        # Вторая капля
        r2 = np.sqrt((X - center2_x) ** 2 + Y ** 2)
        image += intensity2 * np.exp(-(r2 ** 2) / (2 * sigma ** 2))

        # Нормализация
        if image.max() > 0:
            image = image / image.max()

        return image

    def generate_measurement(self, image):
        """Генерация измерений из изображения"""
        image_vector = image.reshape(-1, 1)
        measurement = self.SM @ image_vector
        measurement_reshaped = measurement.reshape(2, -1)
        return measurement_reshaped

    def tikhonov_reconstruction(self, measurement, mu=0.001, kmax=100):
        """Реконструкция методом Тихонова"""
        from ReconstructionMethod import ReconstructionMethod as RM

        Meas = np.concatenate([measurement[0, :], measurement[1, :]])
        recImage = RM.Tikhonov_underdetermined(self.SM, Meas, mu, kmax)
        recImage = recImage.real.reshape(self.image_shape)

        # Нормализация
        if recImage.max() > 0:
            recImage = recImage / recImage.max()

        return recImage

    def cnn_reconstruction(self, measurement):
        """Реконструкция с помощью CNN"""
        if self.cnn_trainer is None:
            raise ValueError("CNN модель не установлена")

        # Предсказание
        reconstructed = self.cnn_trainer.predict(measurement)

        # Изменение размера к исходному
        reconstructed = reconstructed[0, 0, :, :]  # Убираем batch и channel dimensions

        # Обрезаем или интерполируем к нужному размеру
        if reconstructed.shape != self.image_shape:
            # Если размер больше - обрезаем
            if reconstructed.shape[0] > self.nx:
                start_x = (reconstructed.shape[0] - self.nx) // 2
                reconstructed = reconstructed[start_x:start_x + self.nx, :]
            if reconstructed.shape[1] > self.ny:
                start_y = (reconstructed.shape[1] - self.ny) // 2
                reconstructed = reconstructed[:, start_y:start_y + self.ny]

            # Если размер меньше - интерполируем
            if reconstructed.shape[0] < self.nx or reconstructed.shape[1] < self.ny:
                from scipy import ndimage
                reconstructed = ndimage.zoom(reconstructed,
                                             (self.nx / reconstructed.shape[0],
                                              self.ny / reconstructed.shape[1]),
                                             order=1)

        # Нормализация
        if reconstructed.max() > 0:
            reconstructed = reconstructed / reconstructed.max()

        return reconstructed

    def calculate_metrics(self, original, reconstructed, method_name):
        """Вычисление метрик качества"""
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr

        # Нормализация изображений
        orig_norm = (original - original.min()) / (original.max() - original.min() + 1e-10)
        recon_norm = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min() + 1e-10)

        # SSIM
        ssim_value = ssim(orig_norm, recon_norm, data_range=1.0,
                          win_size=7, channel_axis=None)

        # PSNR
        psnr_value = psnr(orig_norm, recon_norm, data_range=1.0)

        # Пространственное разрешение (FWHM)
        fwhm_value = self.calculate_fwhm(reconstructed)

        # MSE
        mse_value = np.mean((orig_norm - recon_norm) ** 2)

        metrics = {
            'method': method_name,
            'ssim': ssim_value,
            'psnr': psnr_value,
            'fwhm': fwhm_value,
            'mse': mse_value
        }

        self.metrics_history.append(metrics)

        return metrics

    def calculate_fwhm(self, image):
        """Вычисление FWHM для оценки разрешения"""
        # Находим максимальное значение
        max_val = np.max(image)
        if max_val == 0:
            return 0

        half_max = max_val / 2

        # Бинаризация по половине максимума
        binary = image > half_max

        if not np.any(binary):
            return 0

        # Находим связные компоненты
        from scipy import ndimage
        labeled, num_features = ndimage.label(binary)

        if num_features == 0:
            return 0

        # Для каждой компоненты вычисляем FWHM
        fwhm_values = []
        for i in range(1, num_features + 1):
            component = labeled == i

            # Находим bounding box
            rows = np.any(component, axis=1)
            cols = np.any(component, axis=0)

            if np.any(rows) and np.any(cols):
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]

                # FWHM как размер bounding box
                fwhm_h = cmax - cmin
                fwhm_v = rmax - rmin
                fwhm_values.extend([fwhm_h, fwhm_v])

        return np.mean(fwhm_values) if fwhm_values else 0

    def compare_methods(self, radius=0.2, distance=0.2):
        """Сравнение методов для конкретного случая"""
        print(f"\nСравнение методов для radius={radius}, distance={distance}")
        print("-" * 50)

        # Генерация тестового случая
        original_image = self.generate_test_case(radius, distance)
        measurement = self.generate_measurement(original_image)

        # Реконструкция методом Тихонова
        print("Реконструкция методом Тихонова...")
        tikhonov_start = time.time()
        tikhonov_recon = self.tikhonov_reconstruction(measurement)
        tikhonov_time = time.time() - tikhonov_start

        # Реконструкция методом CNN
        print("Реконструкция методом CNN...")
        cnn_start = time.time()
        cnn_recon = self.cnn_reconstruction(measurement)
        cnn_time = time.time() - cnn_start

        # Вычисление метрик
        tikhonov_metrics = self.calculate_metrics(original_image, tikhonov_recon, 'Тихонов')
        cnn_metrics = self.calculate_metrics(original_image, cnn_recon, 'CNN')

        # Добавляем время
        tikhonov_metrics['time'] = tikhonov_time
        cnn_metrics['time'] = cnn_time

        # Вывод результатов
        print(f"\nРезультаты для radius={radius}, distance={distance}:")
        print(f"{'Метод':<10} {'SSIM':<8} {'PSNR (дБ)':<12} {'FWHM':<8} {'Время (с)':<10}")
        print("-" * 50)
        print(f"{'Тихонов':<10} {tikhonov_metrics['ssim']:<8.4f} {tikhonov_metrics['psnr']:<12.2f} "
              f"{tikhonov_metrics['fwhm']:<8.2f} {tikhonov_metrics['time']:<10.4f}")
        print(f"{'CNN':<10} {cnn_metrics['ssim']:<8.4f} {cnn_metrics['psnr']:<12.2f} "
              f"{cnn_metrics['fwhm']:<8.2f} {cnn_metrics['time']:<10.4f}")

        # Визуализация
        self.visualize_comparison(original_image, tikhonov_recon, cnn_recon,
                                  radius, distance, tikhonov_metrics, cnn_metrics)

        return {
            'original': original_image,
            'tikhonov': tikhonov_recon,
            'cnn': cnn_recon,
            'metrics': {
                'tikhonov': tikhonov_metrics,
                'cnn': cnn_metrics
            }
        }

    def visualize_comparison(self, original, tikhonov, cnn, radius, distance,
                             tikhonov_metrics, cnn_metrics):
        """Визуализация сравнения методов"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Определяем общие пределы для цветовой шкалы
        vmin = 0
        vmax = 1

        # Оригинальное изображение
        im0 = axes[0, 0].imshow(original, cmap='hot', vmin=vmin, vmax=vmax)
        axes[0, 0].set_title('Оригинальное изображение')
        axes[0, 0].set_xlabel(f'radius={radius}, distance={distance}')
        plt.colorbar(im0, ax=axes[0, 0])

        # Реконструкция Тихонова
        im1 = axes[0, 1].imshow(tikhonov, cmap='hot', vmin=vmin, vmax=vmax)
        axes[0, 1].set_title('Метод Тихонова')
        axes[0, 1].set_xlabel(f"SSIM: {tikhonov_metrics['ssim']:.4f}\n"
                              f"PSNR: {tikhonov_metrics['psnr']:.2f} дБ\n"
                              f"FWHM: {tikhonov_metrics['fwhm']:.2f} пикс.")
        plt.colorbar(im1, ax=axes[0, 1])

        # Реконструкция CNN
        im2 = axes[0, 2].imshow(cnn, cmap='hot', vmin=vmin, vmax=vmax)
        axes[0, 2].set_title('CNN метод')
        axes[0, 2].set_xlabel(f"SSIM: {cnn_metrics['ssim']:.4f}\n"
                              f"PSNR: {cnn_metrics['psnr']:.2f} дБ\n"
                              f"FWHM: {cnn_metrics['fwhm']:.2f} пикс.")
        plt.colorbar(im2, ax=axes[0, 2])

        # Профили интенсивности по горизонтали
        center_y = original.shape[0] // 2
        x = np.linspace(-1, 1, original.shape[1])

        axes[1, 0].plot(x, original[center_y, :], 'k-', linewidth=2, label='Оригинал')
        axes[1, 0].plot(x, tikhonov[center_y, :], 'r--', linewidth=1.5, label='Тихонов')
        axes[1, 0].plot(x, cnn[center_y, :], 'b:', linewidth=1.5, label='CNN')
        axes[1, 0].set_xlabel('X координата')
        axes[1, 0].set_ylabel('Интенсивность')
        axes[1, 0].set_title('Горизонтальный профиль через центр')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Разность изображений
        diff_tikhonov = np.abs(original - tikhonov)
        diff_cnn = np.abs(original - cnn)

        im3 = axes[1, 1].imshow(diff_tikhonov, cmap='Reds', vmin=0, vmax=0.5)
        axes[1, 1].set_title('Ошибка: Тихонов')
        axes[1, 1].set_xlabel(f'MSE: {tikhonov_metrics["mse"]:.6f}')
        plt.colorbar(im3, ax=axes[1, 1])

        im4 = axes[1, 2].imshow(diff_cnn, cmap='Reds', vmin=0, vmax=0.5)
        axes[1, 2].set_title('Ошибка: CNN')
        axes[1, 2].set_xlabel(f'MSE: {cnn_metrics["mse"]:.6f}')
        plt.colorbar(im4, ax=axes[1, 2])

        plt.suptitle(f'Сравнение методов реконструкции MPI (radius={radius}, distance={distance})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Сохранение
        filename = f'./DATA/results/comparison_r{radius}_d{distance}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()

    def summary_table(self, all_results):
        """Вывод сводной таблицы результатов"""
        print("\n" + "=" * 80)
        print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
        print("=" * 80)

        print(f"{'Radius':<8} {'Distance':<10} {'Метод':<10} {'SSIM':<8} {'PSNR (дБ)':<12} "
              f"{'FWHM':<8} {'Время (с)':<10}")
        print("-" * 80)

        for item in all_results:
            radius = item['radius']
            distance = item['distance']
            results = item['results']

            tik_metrics = results['metrics']['tikhonov']
            cnn_metrics = results['metrics']['cnn']

            print(f"{radius:<8.3f} {distance:<10.3f} {'Тихонов':<10} "
                  f"{tik_metrics['ssim']:<8.4f} {tik_metrics['psnr']:<12.2f} "
                  f"{tik_metrics['fwhm']:<8.2f} {tik_metrics['time']:<10.4f}")

            print(f"{'':<8} {'':<10} {'CNN':<10} "
                  f"{cnn_metrics['ssim']:<8.4f} {cnn_metrics['psnr']:<12.2f} "
                  f"{cnn_metrics['fwhm']:<8.2f} {cnn_metrics['time']:<10.4f}")
            print("-" * 80)

        # Статистика
        print("\n" + "=" * 80)
        print("СТАТИСТИКА ПО ВСЕМ ЭКСПЕРИМЕНТАМ")
        print("=" * 80)

        if len(all_results) == 0:
            print("Нет результатов для статистики")
            return

        tik_ssim_avg = np.mean([r['results']['metrics']['tikhonov']['ssim'] for r in all_results])
        cnn_ssim_avg = np.mean([r['results']['metrics']['cnn']['ssim'] for r in all_results])

        tik_psnr_avg = np.mean([r['results']['metrics']['tikhonov']['psnr'] for r in all_results])
        cnn_psnr_avg = np.mean([r['results']['metrics']['cnn']['psnr'] for r in all_results])

        tik_time_avg = np.mean([r['results']['metrics']['tikhonov']['time'] for r in all_results])
        cnn_time_avg = np.mean([r['results']['metrics']['cnn']['time'] for r in all_results])

        print(f"Средний SSIM:")
        print(f"  Тихонов: {tik_ssim_avg:.4f}")
        print(f"  CNN:     {cnn_ssim_avg:.4f}")
        if tik_ssim_avg > 0:
            print(f"  Улучшение: +{(cnn_ssim_avg - tik_ssim_avg):.4f} ({(cnn_ssim_avg / tik_ssim_avg - 1) * 100:.1f}%)")

        print(f"\nСредний PSNR:")
        print(f"  Тихонов: {tik_psnr_avg:.2f} дБ")
        print(f"  CNN:     {cnn_psnr_avg:.2f} дБ")
        print(f"  Улучшение: +{(cnn_psnr_avg - tik_psnr_avg):.2f} дБ")

        print(f"\nСреднее время реконструкции:")
        print(f"  Тихонов: {tik_time_avg:.4f} с")
        print(f"  CNN:     {cnn_time_avg:.4f} с")
        if cnn_time_avg > 0:
            print(f"  Ускорение: {tik_time_avg / cnn_time_avg:.1f}x")

    def save_results_to_file(self, all_results, filename='./DATA/results/summary_results.txt'):
        """Сохранение результатов в файл"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("РЕЗУЛЬТАТЫ СРАВНЕНИЯ МЕТОДОВ РЕКОНСТРУКЦИИ MPI\n")
            f.write("=" * 80 + "\n\n")

            for item in all_results:
                radius = item['radius']
                distance = item['distance']
                results = item['results']

                tik_metrics = results['metrics']['tikhonov']
                cnn_metrics = results['metrics']['cnn']

                f.write(f"ЭКСПЕРИМЕНТ: radius={radius}, distance={distance}\n")
                f.write("-" * 60 + "\n")
                f.write(f"{'Метод':<10} {'SSIM':<8} {'PSNR (дБ)':<12} {'FWHM':<8} {'Время (с)':<10}\n")
                f.write(f"{'Тихонов':<10} {tik_metrics['ssim']:<8.4f} {tik_metrics['psnr']:<12.2f} "
                        f"{tik_metrics['fwhm']:<8.2f} {tik_metrics['time']:<10.4f}\n")
                f.write(f"{'CNN':<10} {cnn_metrics['ssim']:<8.4f} {cnn_metrics['psnr']:<12.2f} "
                        f"{cnn_metrics['fwhm']:<8.2f} {cnn_metrics['time']:<10.4f}\n")

                ssim_improvement = cnn_metrics['ssim'] - tik_metrics['ssim']
                psnr_improvement = cnn_metrics['psnr'] - tik_metrics['psnr']
                time_speedup = tik_metrics['time'] / cnn_metrics['time'] if cnn_metrics['time'] > 0 else 0

                f.write(f"Улучшение CNN: SSIM +{ssim_improvement:.4f}, "
                        f"PSNR +{psnr_improvement:.2f} дБ, "
                        f"Ускорение {time_speedup:.1f}x\n\n")

            # Итоговая статистика
            if len(all_results) > 0:
                f.write("\n" + "=" * 80 + "\n")
                f.write("ОБЩАЯ СТАТИСТИКА\n")
                f.write("=" * 80 + "\n\n")

                tik_ssim_avg = np.mean([r['results']['metrics']['tikhonov']['ssim'] for r in all_results])
                cnn_ssim_avg = np.mean([r['results']['metrics']['cnn']['ssim'] for r in all_results])

                tik_psnr_avg = np.mean([r['results']['metrics']['tikhonov']['psnr'] for r in all_results])
                cnn_psnr_avg = np.mean([r['results']['metrics']['cnn']['psnr'] for r in all_results])

                tik_time_avg = np.mean([r['results']['metrics']['tikhonov']['time'] for r in all_results])
                cnn_time_avg = np.mean([r['results']['metrics']['cnn']['time'] for r in all_results])

                f.write(f"Средний SSIM:\n")
                f.write(f"  Тихонов: {tik_ssim_avg:.4f}\n")
                f.write(f"  CNN:     {cnn_ssim_avg:.4f}\n")
                if tik_ssim_avg > 0:
                    f.write(
                        f"  Улучшение: +{(cnn_ssim_avg - tik_ssim_avg):.4f} ({(cnn_ssim_avg / tik_ssim_avg - 1) * 100:.1f}%)\n\n")

                f.write(f"Средний PSNR:\n")
                f.write(f"  Тихонов: {tik_psnr_avg:.2f} дБ\n")
                f.write(f"  CNN:     {cnn_psnr_avg:.2f} дБ\n")
                f.write(f"  Улучшение: +{(cnn_psnr_avg - tik_psnr_avg):.2f} дБ\n\n")

                f.write(f"Среднее время реконструкции:\n")
                f.write(f"  Тихонов: {tik_time_avg:.4f} с\n")
                f.write(f"  CNN:     {cnn_time_avg:.4f} с\n")
                if cnn_time_avg > 0:
                    f.write(f"  Ускорение: {tik_time_avg / cnn_time_avg:.1f}x\n")

        print(f"\nРезультаты сохранены в файл: {filename}")

    def plot_summary_results(self, all_results):
        """Построение графиков сводных результатов"""
        if not all_results:
            print("Нет данных для построения графиков")
            return

        # Подготовка данных
        distances = [item['distance'] for item in all_results]
        tik_ssim = [item['results']['metrics']['tikhonov']['ssim'] for item in all_results]
        cnn_ssim = [item['results']['metrics']['cnn']['ssim'] for item in all_results]

        tik_psnr = [item['results']['metrics']['tikhonov']['psnr'] for item in all_results]
        cnn_psnr = [item['results']['metrics']['cnn']['psnr'] for item in all_results]

        tik_time = [item['results']['metrics']['tikhonov']['time'] for item in all_results]
        cnn_time = [item['results']['metrics']['cnn']['time'] for item in all_results]

        # Создание графиков
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # График 1: SSIM по расстояниям
        axes[0, 0].plot(distances, tik_ssim, 'ro-', label='Тихонов', linewidth=2, markersize=8)
        axes[0, 0].plot(distances, cnn_ssim, 'bs-', label='CNN', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Расстояние между каплями')
        axes[0, 0].set_ylabel('SSIM')
        axes[0, 0].set_title('Сравнение SSIM для разных расстояний')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].invert_xaxis()  # Большее расстояние слева

        # Добавление значений на график
        for i, (d, tik, cnn) in enumerate(zip(distances, tik_ssim, cnn_ssim)):
            axes[0, 0].text(d, tik + 0.01, f'{tik:.3f}', ha='center', va='bottom', fontsize=8)
            axes[0, 0].text(d, cnn + 0.01, f'{cnn:.3f}', ha='center', va='bottom', fontsize=8)

        # График 2: PSNR по расстояниям
        axes[0, 1].plot(distances, tik_psnr, 'ro-', label='Тихонов', linewidth=2, markersize=8)
        axes[0, 1].plot(distances, cnn_psnr, 'bs-', label='CNN', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Расстояние между каплями')
        axes[0, 1].set_ylabel('PSNR (дБ)')
        axes[0, 1].set_title('Сравнение PSNR для разных расстояний')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].invert_xaxis()

        # График 3: Время выполнения
        bar_width = 0.35
        x_indices = np.arange(len(distances))

        axes[1, 0].bar(x_indices - bar_width / 2, tik_time, bar_width, label='Тихонов', alpha=0.8)
        axes[1, 0].bar(x_indices + bar_width / 2, cnn_time, bar_width, label='CNN', alpha=0.8)
        axes[1, 0].set_xlabel('Расстояние между каплями')
        axes[1, 0].set_ylabel('Время реконструкции (с)')
        axes[1, 0].set_title('Сравнение времени выполнения')
        axes[1, 0].set_xticks(x_indices)
        axes[1, 0].set_xticklabels([f'd={d}' for d in distances])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # График 4: Улучшение CNN
        ssim_improvement = [cnn - tik for tik, cnn in zip(tik_ssim, cnn_ssim)]
        psnr_improvement = [cnn - tik for tik, cnn in zip(tik_psnr, cnn_psnr)]

        x = np.arange(len(distances))
        width = 0.35

        axes[1, 1].bar(x - width / 2, ssim_improvement, width, label='Улучшение SSIM', color='green', alpha=0.7)
        axes[1, 1].bar(x + width / 2, psnr_improvement, width, label='Улучшение PSNR (дБ)', color='orange', alpha=0.7)
        axes[1, 1].set_xlabel('Расстояние между каплями')
        axes[1, 1].set_ylabel('Улучшение')
        axes[1, 1].set_title('Улучшение CNN по сравнению с методом Тихонова')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([f'd={d}' for d in distances])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        plt.suptitle('Сводные результаты сравнения методов реконструкции MPI', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Сохранение графика
        summary_plot_path = './DATA/results/summary_plot.png'
        plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"\nСводный график сохранен: {summary_plot_path}")

    def run_comparison_experiments(self, radius_list=[0.2], distance_list=[0.2, 0.1, 0.05]):
        """Запуск серии экспериментов"""
        print("=" * 60)
        print("СРАВНЕНИЕ МЕТОДОВ РЕКОНСТРУКЦИИ MPI")
        print("=" * 60)

        all_results = []

        for radius in radius_list:
            for distance in distance_list:
                if distance >= 0:  # Простая проверка
                    print(f"\n{'=' * 50}")
                    print(f"Эксперимент: радиус={radius}, расстояние={distance}")
                    print(f"{'=' * 50}")

                    result = self.compare_methods(radius, distance)
                    all_results.append({
                        'radius': radius,
                        'distance': distance,
                        'results': result
                    })
                else:
                    print(f"\nПропускаем radius={radius}, distance={distance}")

        # Сводная таблица результатов
        self.summary_table(all_results)

        # Сохранение результатов в файл
        self.save_results_to_file(all_results)

        # Дополнительная визуализация сводных результатов
        self.plot_summary_results(all_results)

        return all_results


def main():
    """Главная функция для запуска сравнения методов"""

    print("=" * 60)
    print("ПАЙПЛАЙН СРАВНЕНИЯ МЕТОДОВ РЕКОНСТРУКЦИИ MPI")
    print("=" * 60)

    # Создаем директории если их нет
    os.makedirs('./DATA/results', exist_ok=True)
    os.makedirs('./DATA/dataset', exist_ok=True)
    os.makedirs('./DATA/models', exist_ok=True)

    # Шаг 1: Проверка и генерация датасета
    if not os.path.exists('./DATA/dataset/X_train.npy'):
        print("\n1. Генерация датасета для обучения CNN...")
        generator = MPIDatasetGenerator()
        generator.create_dataset(num_samples=2000)
    else:
        print("\n1. Датасет уже существует, загружаем...")

    # Шаг 2: Обучение или загрузка CNN модели
    print("\n2. Обучение CNN модели...")
    if not os.path.exists('./DATA/models/mpi_cnn_best.pth'):
        cnn_trainer = train_cnn_model()
    else:
        print("  Обученная модель уже существует, загружаем...")
        cnn_trainer = MPITrainer()
        cnn_trainer.create_model(input_channels=4)

        # Загрузка сохраненной модели
        if torch.cuda.is_available():
            checkpoint = torch.load('./DATA/models/mpi_cnn_best.pth')
        else:
            checkpoint = torch.load('./DATA/models/mpi_cnn_best.pth',
                                    map_location=torch.device('cpu'))

        cnn_trainer.model.load_state_dict(checkpoint['model_state_dict'])
        cnn_trainer.setup_training()
        print(f"  Модель загружена из эпохи {checkpoint['epoch'] + 1}")

    # Шаг 3: Сравнение методов
    print("\n3. Сравнение методов реконструкции...")

    comparator = MPIReconstructionComparator()
    comparator.set_cnn_model(cnn_trainer)

    # Запуск экспериментов для разных расстояний
    radius = 0.2
    distances = [0.2, 0.1, 0.05]

    print(f"\nТестовые случаи:")
    print(f"  Радиус капель: {radius}")
    print(f"  Расстояния между центрами: {distances}")

    results = comparator.run_comparison_experiments(
        radius_list=[radius],
        distance_list=distances
    )

    # Вывод краткой информации о результатах
    if results:
        print("\n" + "=" * 60)
        print("КРАТКИЕ ИТОГИ:")
        print("=" * 60)

        for result in results:
            radius = result['radius']
            distance = result['distance']
            metrics_tik = result['results']['metrics']['tikhonov']
            metrics_cnn = result['results']['metrics']['cnn']

            print(f"\nРадиус: {radius}, Расстояние: {distance}")
            print(f"  SSIM: {metrics_tik['ssim']:.4f} -> {metrics_cnn['ssim']:.4f} "
                  f"(+{metrics_cnn['ssim'] - metrics_tik['ssim']:.4f})")
            print(f"  PSNR: {metrics_tik['psnr']:.2f} -> {metrics_cnn['psnr']:.2f} дБ "
                  f"(+{metrics_cnn['psnr'] - metrics_tik['psnr']:.2f} дБ)")
            print(f"  Время: {metrics_tik['time']:.4f} -> {metrics_cnn['time']:.4f} с "
                  f"(ускорение {metrics_tik['time'] / metrics_cnn['time']:.1f}x)")

    print("\n" + "=" * 60)
    print("ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН!")
    print("=" * 60)
    print("\nРезультаты сохранены в:")
    print("  ./DATA/results/ - изображения сравнения")
    print("  ./DATA/results/summary_results.txt - текстовый отчет")
    print("  ./DATA/results/summary_plot.png - сводный график")


main()
