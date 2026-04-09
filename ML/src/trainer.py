"""Тренировка нейросетевых моделей"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import os


class MPITrainer:
    """Тренировка CNN для MPI реконструкции"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

    def create_model(self, input_channels=4, output_channels=1):
        """Создание модели"""
        from .models import MPIReconstructionCNN
        self.model = MPIReconstructionCNN(input_channels, output_channels)
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

        if os.path.exists(save_path):
            checkpoint = torch.load(save_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Загружена лучшая модель из эпохи {checkpoint['epoch'] + 1}")

        return train_losses, val_losses

    def predict(self, measurement):
        """Предсказание на новых данных"""
        import numpy as np
        import torch

        self.model.eval()

        with torch.no_grad():
            if isinstance(measurement, np.ndarray):
                real_part = measurement.real
                imag_part = measurement.imag
                combined = np.concatenate([real_part, imag_part], axis=0)

                n_measurements = measurement.shape[1]
                size = int(np.sqrt(n_measurements))
                if size * size != n_measurements:
                    size += 1
                    target_size = size * size

                    padded = np.zeros((4, target_size), dtype=np.float32)
                    padded[:, :n_measurements] = combined
                    combined = padded

                measurement_reshaped = combined.reshape(1, 4, size, size)
                measurement_tensor = torch.tensor(measurement_reshaped, dtype=torch.float32)
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
        print(f"Модель загружена из эпохи {checkpoint['epoch'] + 1}")
        return checkpoint