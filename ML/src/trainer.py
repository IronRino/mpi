"""Тренировка нейросетевых моделей"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import os
import numpy as np


class MPITrainer:
    """Базовый тренер для MPI моделей"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"MPITrainer: Используется устройство: {self.device}")
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.model_type = None  # Добавляем атрибут для хранения типа модели

    def create_model(self, model_type='cnn', **kwargs):
        """Создание модели указанного типа"""
        from .models import MPIReconstructionCNN, MoDLNetwork, DiffusionModel, DiffusionUNet, CombinedHybridModel

        self.model_type = model_type  # Сохраняем тип модели

        if model_type == 'cnn':
            self.model = MPIReconstructionCNN(
                input_channels=kwargs.get('input_channels', 4),
                output_channels=kwargs.get('output_channels', 1)
            )
        elif model_type == 'modl':
            self.model = MoDLNetwork(
                system_matrix=kwargs.get('system_matrix'),
                image_shape=kwargs.get('image_shape'),
                n_iterations=kwargs.get('n_iterations', 5),
                lambda_param=kwargs.get('lambda_param', 0.01)
            )
        elif model_type == 'diffusion':
            denoiser = DiffusionUNet(in_channels=2, out_channels=1, image_size=51)
            self.model = DiffusionModel(
                denoiser=denoiser,
                n_steps=kwargs.get('n_steps', 1000)
            )
        elif model_type == 'hybrid':
            self.model = CombinedHybridModel(
                system_matrix=kwargs.get('system_matrix'),
                image_shape=kwargs.get('image_shape'),
                modl_iterations=kwargs.get('modl_iterations', 3),
                diffusion_steps=kwargs.get('diffusion_steps', 50)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model.to(self.device)
        return self.model

    def setup_training(self, learning_rate=1e-3, weight_decay=1e-5):
        """Настройка обучения"""
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate,
                                          weight_decay=weight_decay)
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

            model_type_lower = str(self.model_type).lower() if self.model_type else ''

            if 'diffusion' in model_type_lower:
                # Для диффузионной модели:
                # measurements имеет форму (B, 4, H, W) - 4 канала
                # targets имеет форму (B, 1, H, W) - 1 канал

                # Изменяем размер измерений под целевое изображение если нужно
                if measurements.shape[-2:] != targets.shape[-2:]:
                    measurements = F.interpolate(measurements, size=targets.shape[-2:],
                                                 mode='bilinear', align_corners=False)

                # Усредняем каналы измерений до 1 канала для использования как условие
                # (или можно использовать как есть, но модель ожидает 1 канал условия)
                condition = measurements.mean(dim=1, keepdim=True)  # (B, 1, H, W)

                outputs = self.model(targets, condition=condition)
                if isinstance(outputs, tuple):
                    predicted_noise, true_noise = outputs
                    loss = self.criterion(predicted_noise, true_noise)
                else:
                    loss = self.criterion(outputs, targets)
            else:
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

                model_type_lower = str(self.model_type).lower() if self.model_type else ''

                if 'diffusion' in model_type_lower:
                    # Для диффузионной модели
                    if measurements.shape[-2:] != targets.shape[-2:]:
                        measurements = F.interpolate(measurements, size=targets.shape[-2:],
                                                     mode='bilinear', align_corners=False)

                    # Усредняем каналы измерений до 1 канала
                    condition = measurements.mean(dim=1, keepdim=True)  # (B, 1, H, W)

                    outputs = self.model(targets, condition=condition)
                    if isinstance(outputs, tuple):
                        predicted_noise, true_noise = outputs
                        loss = self.criterion(predicted_noise, true_noise)
                    else:
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(measurements)
                    loss = self.criterion(outputs, targets)

                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader, epochs=50, save_path='./DATA/models/mpi_model_best.pth'):
        """Полный цикл обучения"""
        print(f"Начало обучения модели...")

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

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'model_type': self.model_type,  # Сохраняем тип модели
                }, save_path)
                print(f"  Сохранена лучшая модель (epoch {epoch + 1}, val_loss: {val_loss:.6f})")

            epoch_time = time.time() - start_time

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, "
                      f"Time: {epoch_time:.1f}s")

        if os.path.exists(save_path):
            checkpoint = torch.load(save_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model_type = checkpoint.get('model_type', self.model_type)
            print(f"Загружена лучшая модель из эпохи {checkpoint['epoch'] + 1}")

        return train_losses, val_losses

    def predict(self, measurement):
        """Предсказание на новых данных"""
        import numpy as np
        import torch

        self.model.eval()

        with torch.no_grad():
            if isinstance(measurement, np.ndarray):
                # Для CNN модели вход должен быть 4D: (B, C, H, W)
                if self.model_type == 'cnn':
                    # Измерения имеют форму (2, n_measurements) - комплексные
                    # Нужно преобразовать в формат, который ожидает CNN

                    # Разделяем на реальную и мнимую части
                    real_part = measurement.real  # (2, n_measurements)
                    imag_part = measurement.imag  # (2, n_measurements)

                    # Объединяем в 4 канала: real1, imag1, real2, imag2
                    combined = np.concatenate([real_part, imag_part], axis=0)  # (4, n_measurements)

                    # Находим квадратный размер
                    n_measurements = combined.shape[1]
                    size = int(np.sqrt(n_measurements))
                    if size * size != n_measurements:
                        size += 1
                        target_size = size * size
                        padded = np.zeros((4, target_size), dtype=np.float32)
                        padded[:, :n_measurements] = combined
                        combined = padded

                    # Преобразуем в квадратную форму (4, size, size)
                    measurement_tensor = combined.reshape(1, 4, size, size)
                    measurement_tensor = torch.tensor(measurement_tensor, dtype=torch.float32)

                elif self.model_type == 'modl':
                    # Для MoDL вход - вектор измерений
                    real_part = measurement.real
                    imag_part = measurement.imag
                    meas_vector = np.concatenate([real_part.flatten(), imag_part.flatten()])
                    measurement_tensor = torch.tensor(meas_vector, dtype=torch.float32).unsqueeze(0)

                else:
                    # По умолчанию - вектор
                    if measurement.ndim == 2 and measurement.shape[0] == 2:
                        real_part = measurement.real
                        imag_part = measurement.imag
                        meas_vector = np.concatenate([real_part.flatten(), imag_part.flatten()])
                    else:
                        meas_vector = measurement.flatten()
                    measurement_tensor = torch.tensor(meas_vector, dtype=torch.float32).unsqueeze(0)

            else:
                measurement_tensor = measurement

            measurement_tensor = measurement_tensor.to(self.device)
            output = self.model(measurement_tensor)

        return output.cpu().numpy()

    def load_model(self, path):
        """Загрузка сохраненной модели"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if hasattr(self, 'optimizer') and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model_type = checkpoint.get('model_type', self.model_type)
        print(f"Модель загружена из эпохи {checkpoint['epoch'] + 1}")
        return checkpoint


class ModelTrainerFactory:
    """Фабрика для создания тренеров разных моделей"""

    @staticmethod
    def create_cnn_trainer(input_channels=4, output_channels=1, learning_rate=1e-3):
        trainer = MPITrainer()
        trainer.create_model('cnn', input_channels=input_channels, output_channels=output_channels)
        trainer.setup_training(learning_rate=learning_rate)
        return trainer

    @staticmethod
    def create_modl_trainer(system_matrix, image_shape, n_iterations=5, lambda_param=0.01, learning_rate=1e-3):
        trainer = MPITrainer()
        trainer.create_model('modl',
                             system_matrix=system_matrix,
                             image_shape=image_shape,
                             n_iterations=n_iterations,
                             lambda_param=lambda_param)
        trainer.setup_training(learning_rate=learning_rate)
        return trainer

    @staticmethod
    def create_diffusion_trainer(n_steps=1000, learning_rate=1e-4, image_size=51):
        trainer = MPITrainer()
        trainer.create_model('diffusion', n_steps=n_steps, image_size=image_size)
        trainer.setup_training(learning_rate=learning_rate)
        return trainer

    @staticmethod
    def create_hybrid_trainer(system_matrix, image_shape, modl_iterations=3, diffusion_steps=50, learning_rate=1e-4):
        trainer = MPITrainer()
        trainer.create_model('hybrid',
                             system_matrix=system_matrix,
                             image_shape=image_shape,
                             modl_iterations=modl_iterations,
                             diffusion_steps=diffusion_steps)
        trainer.setup_training(learning_rate=learning_rate)
        return trainer