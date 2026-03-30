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


# Генерация датасета
if __name__ == '__main__':
    generator = MPIDatasetGenerator()
    X_train, X_test, y_train, y_test = generator.create_dataset(num_samples=3000)