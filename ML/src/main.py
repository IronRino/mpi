"""Главный модуль для запуска пайплайна со всеми моделями"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from .dataset_generator import MPIDatasetGenerator, MPIDataset
from .trainer import ModelTrainerFactory, MPITrainer
from .comparator import MPIReconstructionComparator
from .models import (
    ChaeSingleLayerNN, DeepImagePrior, ShangCNN, PGNet, DEQMPI,
    KatsMarcAlgorithm, TikhonovReconstructor
)
from .visualization import Visualization


class MoDLDataset:
    """Dataset для MoDL модели"""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        meas = self.X[idx]
        real_part = meas.real
        imag_part = meas.imag
        meas_vector = np.concatenate([real_part.flatten(), imag_part.flatten()])
        target = self.y[idx]
        return (torch.tensor(meas_vector, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32).unsqueeze(0))


class DiffusionDataset:
    """Dataset для диффузионной модели"""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        meas = self.X[idx]
        target = self.y[idx]

        # Преобразование измерений в 4-канальное изображение
        real_part = meas.real
        imag_part = meas.imag
        combined = np.concatenate([real_part, imag_part], axis=0)

        # Изменение размера до квадратной формы
        n_measurements = combined.shape[1]
        size = int(np.sqrt(n_measurements))
        if size * size != n_measurements:
            size += 1
            target_size = size * size
            padded = np.zeros((4, target_size), dtype=np.float32)
            padded[:, :n_measurements] = combined
            combined = padded

        combined = combined.reshape(4, size, size)

        # Интерполяция до нужного размера
        from scipy import ndimage
        if combined.shape[1] != 51:
            combined = ndimage.zoom(combined, (1, 51 / combined.shape[1], 51 / combined.shape[2]), order=1)

        return (torch.tensor(combined, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32).unsqueeze(0))


def create_directories():
    """Создание необходимых директорий"""
    os.makedirs('./DATA/dataset', exist_ok=True)
    os.makedirs('./DATA/models', exist_ok=True)
    os.makedirs('./DATA/results', exist_ok=True)
    os.makedirs('./DATA/results/training_curves', exist_ok=True)


def generate_dataset(num_samples=5000):
    """Генерация датасета"""
    print("\n" + "="*70)
    print("1. ГЕНЕРАЦИЯ ДАТАСЕТА")
    print("="*70)
    generator = MPIDatasetGenerator()
    generator.create_dataset(num_samples=num_samples)
    return generator


def train_cnn_model():
    """Обучение улучшенной CNN модели"""
    print("\n" + "=" * 70)
    print("2. ОБУЧЕНИЕ CNN МОДЕЛИ")
    print("=" * 70)

    train_dataset = MPIDataset('./DATA/dataset/X_train.npy', './DATA/dataset/y_train.npy')
    test_dataset = MPIDataset('./DATA/dataset/X_test.npy', './DATA/dataset/y_test.npy')

    sample_measurement, sample_target = train_dataset[0]
    print(f"  Размер входных данных: {sample_measurement.shape}")
    print(f"  Размер выходных данных: {sample_target.shape}")

    # Проверка согласованности размеров
    assert sample_measurement.shape[0] == 4, f"Expected 4 input channels, got {sample_measurement.shape[0]}"
    print(f"  ✓ Входные каналы: {sample_measurement.shape[0]}")
    print(f"  ✓ Входной размер: {sample_measurement.shape[1]}x{sample_measurement.shape[2]}")
    print(f"  ✓ Выходной размер: {sample_target.shape[1]}x{sample_target.shape[2]}")

    # Используем меньший batch size для стабильности
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    trainer = ModelTrainerFactory.create_cnn_trainer(
        input_channels=4,
        output_channels=1,
        learning_rate=1e-3,
        base_filters=32  # Уменьшаем для начала для экономии памяти
    )

    # Обучаем меньше эпох для теста
    train_losses, val_losses = trainer.train(
        train_loader, val_loader,
        epochs=30,  # Уменьшаем для быстрого теста
        save_path='./DATA/models/cnn_best.pth'
    )

    # Сохранение кривых обучения
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('CNN Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./DATA/results/training_curves/cnn_training.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ CNN модель обучена")
    return trainer


def train_modl_model(system_matrix, image_shape):
    """Обучение улучшенной MoDL модели"""
    print("\n" + "=" * 70)
    print("3. ОБУЧЕНИЕ MoDL МОДЕЛИ")
    print("=" * 70)

    X_train = np.load('./DATA/dataset/X_train.npy')
    y_train = np.load('./DATA/dataset/y_train.npy')

    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")

    # Используем только часть данных для ускорения (опционально)
    n_samples = min(2000, len(X_train))
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    X_train = X_train[indices]
    y_train = y_train[indices]

    print(f"  Используется {n_samples} образцов для обучения")

    split_idx = int(0.8 * len(X_train))
    train_dataset = MoDLDataset(X_train[:split_idx], y_train[:split_idx])
    val_dataset = MoDLDataset(X_train[split_idx:], y_train[split_idx:])

    # Для Windows важно установить num_workers=0
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    trainer = ModelTrainerFactory.create_modl_trainer(
        system_matrix=system_matrix,
        image_shape=image_shape,
        n_iterations=3,  # Уменьшаем для ускорения
        lambda_param=0.01,
        learning_rate=1e-3,
        base_filters=32  # Уменьшаем для экономии памяти
    )

    train_losses, val_losses = trainer.train(
        train_loader, val_loader,
        epochs=20,  # Уменьшаем для быстрого теста
        save_path='./DATA/models/modl_best.pth'
    )

    # Сохранение кривых обучения
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('MoDL Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./DATA/results/training_curves/modl_training.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ MoDL модель обучена")
    return trainer


def train_diffusion_model():
    """Обучение улучшенной диффузионной модели"""
    print("\n" + "=" * 70)
    print("4. ОБУЧЕНИЕ DIFFUSION МОДЕЛИ")
    print("=" * 70)

    X_train = np.load('./DATA/dataset/X_train.npy')
    y_train = np.load('./DATA/dataset/y_train.npy')

    # Используем меньше данных для ускорения
    n_samples = min(500, len(X_train))
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    X_train = X_train[indices]
    y_train = y_train[indices]

    print(f"  Используется {n_samples} образцов для обучения")

    # Подготовка данных
    X_train_prepared = []
    y_train_prepared = []

    print("  Подготовка данных...")
    for i in tqdm(range(len(X_train)), desc="  Подготовка"):
        meas = X_train[i]
        target = y_train[i]

        # Преобразование измерений в 4-канальное изображение
        real_part = meas.real
        imag_part = meas.imag
        combined = np.concatenate([real_part, imag_part], axis=0)

        # Изменение размера до квадратной формы
        n_measurements = combined.shape[1]
        size = int(np.ceil(np.sqrt(n_measurements)))
        target_size = size * size
        padded = np.zeros((4, target_size), dtype=np.float32)
        padded[:, :n_measurements] = combined
        combined = padded.reshape(4, size, size)

        # Изменение размера до 51x51
        from scipy import ndimage
        if combined.shape[1] != 51:
            combined = ndimage.zoom(combined, (1, 51 / combined.shape[1], 51 / combined.shape[2]), order=1)

        X_train_prepared.append(combined)
        y_train_prepared.append(target)

    X_train_prepared = np.array(X_train_prepared, dtype=np.float32)
    y_train_prepared = np.array(y_train_prepared, dtype=np.float32)

    # Добавляем канал для target
    y_train_prepared = y_train_prepared[:, np.newaxis, :, :]

    print(f"  X_train_prepared shape: {X_train_prepared.shape}")
    print(f"  y_train_prepared shape: {y_train_prepared.shape}")

    # Разделение на train/val
    split_idx = int(0.8 * len(X_train_prepared))

    train_dataset = TensorDataset(
        torch.tensor(X_train_prepared[:split_idx], dtype=torch.float32),
        torch.tensor(y_train_prepared[:split_idx], dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_train_prepared[split_idx:], dtype=torch.float32),
        torch.tensor(y_train_prepared[split_idx:], dtype=torch.float32)
    )

    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Параметры модели
    n_steps = 100
    base_filters = 64
    beta_start = 1e-4
    beta_end = 0.02

    trainer = ModelTrainerFactory.create_diffusion_trainer(
        n_steps=n_steps,
        learning_rate=1e-4,
        image_size=51,
        base_filters=base_filters,
        beta_start=beta_start,
        beta_end=beta_end
    )

    train_losses, val_losses = trainer.train(
        train_loader, val_loader,
        epochs=10,
        save_path='./DATA/models/diffusion_best.pth'
    )

    # После обучения сохраняем параметры модели
    checkpoint = torch.load('./DATA/models/diffusion_best.pth')
    checkpoint['n_steps'] = n_steps
    checkpoint['base_filters'] = base_filters
    checkpoint['beta_start'] = beta_start
    checkpoint['beta_end'] = beta_end
    torch.save(checkpoint, './DATA/models/diffusion_best.pth')

    # Сохранение кривых обучения
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Diffusion Model Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./DATA/results/training_curves/diffusion_training.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Diffusion модель обучена (n_steps={n_steps})")
    return trainer


def train_chae_model(system_matrix, image_shape):
    """Обучение модели Chae (2017)"""
    print("\n" + "="*70)
    print("5. ОБУЧЕНИЕ CHAE МОДЕЛИ (2017)")
    print("="*70)

    X_train = np.load('./DATA/dataset/X_train.npy')
    y_train = np.load('./DATA/dataset/y_train.npy')

    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")

    n_measurements = X_train.shape[2]  # 1275
    input_dim = n_measurements * 4  # 5100
    output_dim = image_shape[0] * image_shape[1]  # 2601

    print(f"  input_dim = {input_dim}")
    print(f"  output_dim = {output_dim}")

    # Используем ВСЮ выборку
    X_train_vec = []
    print("  Подготовка данных...")
    for meas in tqdm(X_train, desc="  Преобразование измерений"):
        real_part = meas.real
        imag_part = meas.imag
        meas_vector = np.concatenate([real_part[0], real_part[1], imag_part[0], imag_part[1]])
        X_train_vec.append(meas_vector)

    X_train_vec = np.array(X_train_vec, dtype=np.float32)
    y_train_flat = y_train.reshape(y_train.shape[0], -1)

    print(f"  X_train_vec shape: {X_train_vec.shape}")
    print(f"  y_train_flat shape: {y_train_flat.shape}")

    # Создание модели с увеличенным hidden_dim
    model = ChaeSingleLayerNN(input_dim, output_dim, hidden_dim=1024)

    # Подготовка DataLoader
    dataset = TensorDataset(
        torch.tensor(X_train_vec, dtype=torch.float32),
        torch.tensor(y_train_flat, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    # Обучение
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print(f"  Начало обучения Chae модели на {device}...")
    model.train()
    best_loss = float('inf')
    train_losses = []

    epoch_pbar = tqdm(range(100), desc="Обучение Chae")
    for epoch in epoch_pbar:
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)

        epoch_pbar.set_postfix({'loss': f'{avg_loss:.6f}'})

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), './DATA/models/chae_best.pth')

    # Сохранение кривой обучения
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Chae Model Training Progress')
    plt.grid(True, alpha=0.3)
    plt.savefig('./DATA/results/training_curves/chae_training.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Chae модель обучена, лучшая loss: {best_loss:.6f}")
    return model


def train_shang_model():
    """Обучение модели Shang et al. (2020)"""
    print("\n" + "="*70)
    print("6. ОБУЧЕНИЕ SHANG МОДЕЛИ (2020)")
    print("="*70)

    X_train = np.load('./DATA/dataset/X_train.npy')
    y_train = np.load('./DATA/dataset/y_train.npy')

    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Используется устройство: {device}")

    # Создаем модель с увеличенными фильтрами
    model = ShangCNN(input_channels=1, output_channels=1, base_filters=64)
    model = model.to(device)

    # Подготовка данных с улучшенной начальной реконструкцией
    X_train_images = []
    print("  Подготовка данных...")

    # Загружаем системную матрицу для улучшенной начальной реконструкции
    SM = np.load('./DATA/dataset/SM.npy')

    for meas in tqdm(X_train[:2000], desc="  Обработка измерений"):  # Используем 2000 для скорости
        real_part = meas.real
        imag_part = meas.imag
        meas_vector = np.concatenate([real_part.flatten(), imag_part.flatten()])

        # Улучшенная начальная реконструкция через псевдообращение
        try:
            A_pinv = np.linalg.pinv(SM)
            initial_recon = (A_pinv @ meas_vector).reshape(51, 51)
        except:
            initial_recon = np.zeros((51, 51))

        # Нормализация
        if initial_recon.max() > 0:
            initial_recon = initial_recon / initial_recon.max()

        X_train_images.append(initial_recon)

    X_train_images = np.array(X_train_images)[:, np.newaxis, :, :].astype(np.float32)
    y_train_images = y_train[:2000][:, np.newaxis, :, :].astype(np.float32)

    print(f"  X_train_images shape: {X_train_images.shape}")

    # DataLoader
    dataset = TensorDataset(
        torch.tensor(X_train_images, dtype=torch.float32),
        torch.tensor(y_train_images, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    # Обучение
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    print("  Начало обучения Shang модели...")
    model.train()
    best_loss = float('inf')
    train_losses = []

    epoch_pbar = tqdm(range(50), desc="Обучение Shang")
    for epoch in epoch_pbar:
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        train_losses.append(avg_loss)
        scheduler.step()

        epoch_pbar.set_postfix({'loss': f'{avg_loss:.6f}'})

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), './DATA/models/shang_best.pth')

    # Сохранение кривой обучения
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Shang Model Training Progress')
    plt.grid(True, alpha=0.3)
    plt.savefig('./DATA/results/training_curves/shang_training.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Shang модель обучена, лучшая loss: {best_loss:.6f}")
    return model


def train_dip_model(image_shape):
    """Обучение Deep Image Prior (Dittmer et al., 2020)"""
    print("\n" + "="*70)
    print("7. ОБУЧЕНИЕ DIP МОДЕЛИ (2020)")
    print("="*70)

    # Для DIP не требуется предварительное обучение
    # Модель будет обучаться на каждом измерении отдельно
    model = DeepImagePrior(image_shape, latent_dim=100, n_channels=64)

    print("  ✓ DIP модель создана (будет обучаться на каждом тестовом изображении)")
    return model


def train_pgnet_model(system_matrix, image_shape):
    """Обучение модели PGNet (Wu et al., 2023)"""
    print("\n" + "="*70)
    print("8. ОБУЧЕНИЕ PGMET МОДЕЛИ (2023)")
    print("="*70)

    X_train = np.load('./DATA/dataset/X_train.npy')
    y_train = np.load('./DATA/dataset/y_train.npy')

    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")

    n_measurements = X_train.shape[2]
    input_dim = n_measurements * 4
    output_shape = image_shape

    print(f"  input_dim = {input_dim}")
    print(f"  output_shape = {output_shape}")

    # Используем подвыборку для ускорения (2000 образцов)
    n_samples = min(2000, len(X_train))
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    X_train = X_train[indices]
    y_train = y_train[indices]

    print(f"  Используется {n_samples} образцов для обучения")

    # Подготовка данных
    X_train_vec = []
    print("  Подготовка данных...")
    for meas in tqdm(X_train, desc="  Преобразование измерений"):
        real_part = meas.real
        imag_part = meas.imag
        meas_vector = np.concatenate([real_part[0], real_part[1], imag_part[0], imag_part[1]])
        X_train_vec.append(meas_vector)

    X_train_vec = np.array(X_train_vec, dtype=np.float32)
    y_train_flat = y_train.reshape(y_train.shape[0], -1)

    print(f"  X_train_vec shape: {X_train_vec.shape}")
    print(f"  y_train_flat shape: {y_train_flat.shape}")

    # Создание модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Используется устройство: {device}")

    model = PGNet(input_dim, output_shape, hidden_dim=256, num_heads=8)
    model = model.to(device)

    # DataLoader
    dataset = TensorDataset(
        torch.tensor(X_train_vec, dtype=torch.float32),
        torch.tensor(y_train_flat, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    # Обучение
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print("  Начало обучения PGNet модели...")
    model.train()
    best_loss = float('inf')
    train_losses = []

    epoch_pbar = tqdm(range(50), desc="Обучение PGNet")
    for epoch in epoch_pbar:
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            y_pred_flat = y_pred.view(y_pred.shape[0], -1)
            loss = criterion(y_pred_flat, y_batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)

        epoch_pbar.set_postfix({'loss': f'{avg_loss:.6f}'})

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), './DATA/models/pgnet_best.pth')

    # Сохранение кривой обучения
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('PGNet Training Progress')
    plt.grid(True, alpha=0.3)
    plt.savefig('./DATA/results/training_curves/pgnet_training.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ PGNet модель обучена, лучшая loss: {best_loss:.6f}")
    return model


def train_deq_model(system_matrix, image_shape):
    """Обучение модели DEQ-MPI (Güngör et al., 2024)"""
    print("\n" + "="*70)
    print("9. ОБУЧЕНИЕ DEQ-MPI МОДЕЛИ (2024)")
    print("="*70)

    X_train = np.load('./DATA/dataset/X_train.npy')
    y_train = np.load('./DATA/dataset/y_train.npy')

    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")

    # Используем подвыборку для ускорения
    n_samples = min(1000, len(X_train))
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    X_train = X_train[indices]
    y_train = y_train[indices]

    print(f"  Используется {n_samples} образцов для обучения")

    # Подготовка данных
    n_measurements = X_train.shape[2]
    input_dim = n_measurements * 4

    print(f"  input_dim = {input_dim}")

    X_train_vec = []
    print("  Подготовка данных...")
    for meas in tqdm(X_train, desc="  Преобразование измерений"):
        real_part = meas.real
        imag_part = meas.imag
        meas_vector = np.concatenate([real_part.flatten(), imag_part.flatten()])
        X_train_vec.append(meas_vector)

    X_train_vec = np.array(X_train_vec, dtype=np.float32)
    y_train_img = y_train

    print(f"  X_train_vec shape: {X_train_vec.shape}")
    print(f"  y_train_img shape: {y_train_img.shape}")

    # Настройки устройства
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Используется устройство: {device}")

    # Создание модели
    n_iterations = 5
    lambda_param = 0.1

    model = DEQMPI(system_matrix, image_shape, n_iterations=n_iterations, lambda_param=lambda_param)
    model = model.to(device)

    # Подготовка DataLoader
    dataset = TensorDataset(
        torch.tensor(X_train_vec, dtype=torch.float32),
        torch.tensor(y_train_img[:, np.newaxis, :, :], dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    # Обучение
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print(f"  Начало обучения DEQ-MPI модели (итераций={n_iterations})...")
    model.train()
    best_loss = float('inf')
    train_losses = []

    epoch_pbar = tqdm(range(30), desc="Обучение DEQ-MPI")
    for epoch in epoch_pbar:
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            # Forward pass
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)

        epoch_pbar.set_postfix({'loss': f'{avg_loss:.6f}'})

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'n_iterations': n_iterations,
                'lambda_param': lambda_param
            }, './DATA/models/deq_best.pth')

    # Сохранение кривой обучения
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('DEQ-MPI Training Progress')
    plt.grid(True, alpha=0.3)
    plt.savefig('./DATA/results/training_curves/deq_training.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ DEQ-MPI модель обучена, лучшая loss: {best_loss:.6f}")
    return model


def plot_all_training_curves():
    """Построение сводного графика всех кривых обучения"""
    print("\n" + "="*70)
    print("10. ВИЗУАЛИЗАЦИЯ КРИВЫХ ОБУЧЕНИЯ")
    print("="*70)

    # Список всех моделей и их файлов
    models = {
        'CNN': './DATA/results/training_curves/cnn_training.png',
        'MoDL': './DATA/results/training_curves/modl_training.png',
        'Diffusion': './DATA/results/training_curves/diffusion_training.png',
        'Chae': './DATA/results/training_curves/chae_training.png',
        'Shang': './DATA/results/training_curves/shang_training.png',
        'PGNet': './DATA/results/training_curves/pgnet_training.png',
        'DEQ-MPI': './DATA/results/training_curves/deq_training.png'
    }

    print("  Кривые обучения сохранены в ./DATA/results/training_curves/")
    for name, path in models.items():
        if os.path.exists(path):
            print(f"    ✓ {name}: {path}")


def run_full_comparison(cnn_trainer, modl_trainer, diffusion_trainer,
                        chae_model, shang_model, dip_model, pgnet_model, deq_model):
    """Запуск полного сравнения всех методов"""
    print("\n" + "="*70)
    print("11. СРАВНЕНИЕ ВСЕХ МЕТОДОВ РЕКОНСТРУКЦИИ")
    print("="*70)

    comparator = MPIReconstructionComparator('./../Chinese data/SystemMatrix.h5')
    comparator.set_cnn_model(cnn_trainer)
    comparator.set_modl_model(modl_trainer)
    comparator.set_diffusion_model(diffusion_trainer)
    comparator.set_chae_model(chae_model)
    comparator.set_shang_model(shang_model)
    comparator.set_dip_model(dip_model)
    comparator.set_pgnet_model(pgnet_model)
    comparator.set_deq_model(deq_model)

    radius = 0.2
    distances = [0.2, 0.15, 0.1, 0.05, 0.025]

    print(f"\n  Радиус капель: {radius}")
    print(f"  Расстояния между центрами: {distances}")

    results = comparator.run_full_comparison(radius=radius, distances=distances)

    comparator.print_summary_table()
    comparator.save_results_to_file()

    return results


def load_cnn_model():
    """Загрузка сохраненной CNN модели"""
    print("\nЗагрузка CNN модели...")

    # Загружаем сохраненный checkpoint для определения параметров
    model_path = './DATA/models/cnn_best.pth'

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

        # Определяем параметры из сохраненной модели
        # По ошибке видно, что модель была обучена с base_filters=32
        base_filters = 32  # Это значение из ошибки: enc1.0.weight shape [32, 4, 3, 3]

        # Дополнительные параметры можно определить из структуры сохраненной модели
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Определяем input_channels по первому слою
            if 'enc1.0.weight' in state_dict:
                input_channels = state_dict['enc1.0.weight'].shape[1]
            else:
                input_channels = 4

            if 'final_conv.3.weight' in state_dict:
                output_channels = state_dict['final_conv.3.weight'].shape[0]
            else:
                output_channels = 1
        else:
            input_channels = 4
            output_channels = 1
            base_filters = 32

        print(f"  Загружаем модель с параметрами: input_channels={input_channels}, "
              f"output_channels={output_channels}, base_filters={base_filters}")

        # Создаем модель с правильными параметрами
        from .models import MPIReconstructionCNN
        model = MPIReconstructionCNN(
            input_channels=input_channels,
            output_channels=output_channels,
            base_filters=base_filters
        )

        # Создаем trainer
        trainer = MPITrainer()
        trainer.model = model
        trainer.model.to(trainer.device)
        trainer.model_type = 'cnn'
        trainer.setup_training(learning_rate=1e-3)

        # Загружаем state_dict
        model_state_dict = checkpoint['model_state_dict']
        missing_keys, unexpected_keys = trainer.model.load_state_dict(model_state_dict, strict=False)

        if missing_keys:
            print(f"  Предупреждение: отсутствуют ключи: {missing_keys[:3]}...")
        if unexpected_keys:
            print(f"  Предупреждение: неожиданные ключи: {unexpected_keys[:3]}...")

        # Проверяем успешность загрузки
        if len(missing_keys) == 0 and len(unexpected_keys) == 0:
            print("  ✓ CNN модель загружена полностью")
        else:
            print("  ✓ CNN модель загружена (с незначительными отличиями)")

        return trainer
    else:
        print("  ⚠ Файл модели не найден, создаем новую")
        trainer = ModelTrainerFactory.create_cnn_trainer(
            input_channels=4, output_channels=1, base_filters=32
        )
        return trainer


def load_modl_model(system_matrix, image_shape):
    """Загрузка сохраненной MoDL модели"""
    print("\nЗагрузка MoDL модели...")

    model_path = './DATA/models/modl_best.pth'

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

        # Определяем параметры из сохраненной модели
        n_iterations = checkpoint.get('n_iterations', 3)
        lambda_param = checkpoint.get('lambda_param', 0.01)
        base_filters = checkpoint.get('base_filters', 32)

        print(
            f"  Загружаем модель с параметрами: iterations={n_iterations}, lambda={lambda_param}, filters={base_filters}")

        # Создаем trainer с правильными параметрами
        trainer = ModelTrainerFactory.create_modl_trainer(
            system_matrix=system_matrix,
            image_shape=image_shape,
            n_iterations=n_iterations,
            lambda_param=lambda_param,
            learning_rate=1e-3,
            base_filters=base_filters
        )

        # Загружаем веса
        trainer.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("  ✓ MoDL модель загружена")
        return trainer
    else:
        print("  ⚠ Файл модели не найден, создаем новую")
        trainer = ModelTrainerFactory.create_modl_trainer(
            system_matrix=system_matrix,
            image_shape=image_shape,
            n_iterations=3,
            lambda_param=0.01,
            base_filters=32
        )
        return trainer


def load_diffusion_model():
    """Загрузка сохраненной диффузионной модели"""
    print("\nЗагрузка Diffusion модели...")

    model_path = './DATA/models/diffusion_best.pth'

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

        # Определяем параметры из сохраненной модели
        n_steps = checkpoint.get('n_steps', 500)
        base_filters = checkpoint.get('base_filters', 64)

        print(f"  Загружаем модель с параметрами: steps={n_steps}, filters={base_filters}")

        # Создаем trainer с правильными параметрами
        trainer = ModelTrainerFactory.create_diffusion_trainer(
            n_steps=n_steps,
            learning_rate=1e-4,
            image_size=51,
            base_filters=base_filters
        )

        # Загружаем веса
        trainer.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("  ✓ Diffusion модель загружена")
        return trainer
    else:
        print("  ⚠ Файл модели не найден, создаем новую")
        trainer = ModelTrainerFactory.create_diffusion_trainer(
            n_steps=500,
            base_filters=64
        )
        return trainer


def load_chae_model(system_matrix, image_shape):
    """Загрузка модели Chae (2017)"""
    print("\nЗагрузка Chae модели...")

    # Определяем размеры
    X_train = np.load('./DATA/dataset/X_train.npy')
    n_measurements = X_train.shape[2]
    input_dim = n_measurements * 4
    output_dim = image_shape[0] * image_shape[1]

    model_path = './DATA/models/chae_best.pth'

    if os.path.exists(model_path):
        # Создаем модель с правильными размерами
        model = ChaeSingleLayerNN(input_dim, output_dim, hidden_dim=1024)

        checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(checkpoint, strict=False)
        print("  ✓ Chae модель загружена")
        return model
    else:
        print("  ⚠ Файл модели не найден, создаем новую")
        model = ChaeSingleLayerNN(input_dim, output_dim, hidden_dim=1024)
        return model


def load_shang_model():
    """Загрузка модели Shang et al. (2020)"""
    print("\nЗагрузка Shang модели...")

    model_path = './DATA/models/shang_best.pth'

    if os.path.exists(model_path):
        # Создаем модель
        model = ShangCNN(input_channels=1, output_channels=1, base_filters=32)

        checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(checkpoint, strict=False)
        print("  ✓ Shang модель загружена")
        return model
    else:
        print("  ⚠ Файл модели не найден, создаем новую")
        model = ShangCNN(input_channels=1, output_channels=1, base_filters=32)
        return model


def load_pgnet_model(system_matrix, image_shape):
    """Загрузка модели PGNet (Wu et al., 2023)"""
    print("\nЗагрузка PGNet модели...")

    # Определяем размеры
    X_train = np.load('./DATA/dataset/X_train.npy')
    n_measurements = X_train.shape[2]
    input_dim = n_measurements * 4

    model_path = './DATA/models/pgnet_best.pth'

    if os.path.exists(model_path):
        # Создаем модель
        model = PGNet(input_dim, image_shape, hidden_dim=256, num_heads=8)

        checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(checkpoint, strict=False)
        print("  ✓ PGNet модель загружена")
        return model
    else:
        print("  ⚠ Файл модели не найден, создаем новую")
        model = PGNet(input_dim, image_shape, hidden_dim=256, num_heads=8)
        return model


def load_deq_model(system_matrix, image_shape):
    """Загрузка модели DEQ-MPI (Güngör et al., 2024)"""
    print("\nЗагрузка DEQ-MPI модели...")

    model_path = './DATA/models/deq_best.pth'

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

        # Определяем параметры из сохраненной модели
        n_iterations = checkpoint.get('n_iterations', 3)
        lambda_param = checkpoint.get('lambda_param', 0.1)

        print(f"  Загружаем модель с параметрами: iterations={n_iterations}, lambda={lambda_param}")

        # Создаем модель
        model = DEQMPI(system_matrix, image_shape, n_iterations=n_iterations, lambda_param=lambda_param)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("  ✓ DEQ-MPI модель загружена")
        return model
    else:
        print("  ⚠ Файл модели не найден, создаем новую")
        model = DEQMPI(system_matrix, image_shape, n_iterations=3, lambda_param=0.1)
        return model


def run_pipeline(num_samples=5000, train_models=True):
    """Запуск полного пайплайна с обучением всех моделей"""
    print("=" * 70)
    print("ПАЙПЛАЙН СРАВНЕНИЯ МЕТОДОВ РЕКОНСТРУКЦИИ MPI")
    print("=" * 70)
    print("\nДоступные методы:")
    print("  1. Тихонов (1963) - классическая регуляризация")
    print("  2. KatsMarc (1937) - алгоритм Качмажа (ART)")
    print("  3. Chae (2017) - однослойная полносвязная сеть")
    print("  4. DIP (2020) - Deep Image Prior")
    print("  5. Shang (2020) - CNN для улучшения разрешения")
    print("  6. PGNet (2023) - Projection Generation Network")
    print("  7. DEQ-MPI (2024) - Deep Equilibrium Model")
    print("  8. CNN - улучшенная UNet")
    print("  9. MoDL - Model-based Deep Learning")
    print(" 10. Diffusion - диффузионная модель")

    create_directories()

    # Генерация датасета
    if not os.path.exists('./DATA/dataset/X_train.npy') or num_samples > 2000:
        generator = generate_dataset(num_samples)
        system_matrix = generator.SM
        image_shape = generator.image_shape
    else:
        print("\n" + "=" * 70)
        print("1. ДАТАСЕТ УЖЕ СУЩЕСТВУЕТ")
        print("=" * 70)
        generator = MPIDatasetGenerator()
        system_matrix = generator.SM
        image_shape = generator.image_shape

    print(f"\n  Системная матрица: {system_matrix.shape}")
    print(f"  Размер изображения: {image_shape}")

    # Определяем, нужно ли обучать модели
    if train_models:
        print("\n" + "=" * 35)
        print("РЕЖИМ: ОБУЧЕНИЕ НОВЫХ МОДЕЛЕЙ")
        print("=" * 35)

        cnn_trainer = train_cnn_model()
        modl_trainer = train_modl_model(system_matrix, image_shape)
        diffusion_trainer = train_diffusion_model()
        chae_model = train_chae_model(system_matrix, image_shape)
        shang_model = train_shang_model()
        dip_model = train_dip_model(image_shape)
        pgnet_model = train_pgnet_model(system_matrix, image_shape)
        deq_model = train_deq_model(system_matrix, image_shape)

        plot_all_training_curves()
    else:
        print("\n" + "=" * 35)
        print("РЕЖИМ: ЗАГРУЗКА СОХРАНЕННЫХ МОДЕЛЕЙ")
        print("=" * 35)

        # CNN модель
        cnn_trainer = MPITrainer()
        cnn_path = './DATA/models/cnn_best.pth'
        if os.path.exists(cnn_path):
            checkpoint = torch.load(cnn_path, map_location=cnn_trainer.device)
            base_filters = 32
            input_channels = 4
            output_channels = 1
            if 'model_state_dict' in checkpoint:
                if 'enc1.0.weight' in checkpoint['model_state_dict']:
                    input_channels = checkpoint['model_state_dict']['enc1.0.weight'].shape[1]
                if 'final_conv.3.weight' in checkpoint['model_state_dict']:
                    output_channels = checkpoint['model_state_dict']['final_conv.3.weight'].shape[0]

            from .models import MPIReconstructionCNN
            cnn_trainer.model = MPIReconstructionCNN(
                input_channels=input_channels,
                output_channels=output_channels,
                base_filters=base_filters
            )
            cnn_trainer.model.to(cnn_trainer.device)
            cnn_trainer.model_type = 'cnn'
            cnn_trainer.setup_training(learning_rate=1e-3)
            cnn_trainer.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("  CNN модель загружена")
        else:
            print("  CNN модель не найдена, начинаем обучение...")
            cnn_trainer = train_cnn_model()

        # MoDL модель
        modl_trainer = MPITrainer()
        modl_path = './DATA/models/modl_best.pth'
        if os.path.exists(modl_path):
            checkpoint = torch.load(modl_path, map_location=modl_trainer.device)
            n_iterations = checkpoint.get('n_iterations', 3)
            lambda_param = checkpoint.get('lambda_param', 0.01)
            base_filters = checkpoint.get('base_filters', 32)
            modl_trainer.create_model('modl',
                                      system_matrix=system_matrix,
                                      image_shape=image_shape,
                                      n_iterations=n_iterations,
                                      lambda_param=lambda_param,
                                      base_filters=base_filters)
            modl_trainer.setup_training(learning_rate=1e-3)
            modl_trainer.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("  MoDL модель загружена")
        else:
            print("  MoDL модель не найдена, начинаем обучение...")
            modl_trainer = train_modl_model(system_matrix, image_shape)

        # Diffusion модель
        diffusion_trainer = MPITrainer()
        diffusion_path = './DATA/models/diffusion_best.pth'
        if os.path.exists(diffusion_path):
            checkpoint = torch.load(diffusion_path, map_location=diffusion_trainer.device)
            # Определяем параметры из сохраненной модели
            n_steps = checkpoint.get('n_steps', 100)
            base_filters = checkpoint.get('base_filters', 64)
            beta_start = checkpoint.get('beta_start', 1e-4)
            beta_end = checkpoint.get('beta_end', 0.02)

            print(f"  Загрузка Diffusion модели с параметрами: n_steps={n_steps}, base_filters={base_filters}")

            diffusion_trainer.create_model('diffusion',
                                           n_steps=n_steps,
                                           image_size=image_shape[0],
                                           base_filters=base_filters,
                                           beta_start=beta_start,
                                           beta_end=beta_end)
            diffusion_trainer.setup_training(learning_rate=1e-4)
            diffusion_trainer.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            diffusion_trainer.model_type = 'diffusion'
            print("  Diffusion модель загружена")
        else:
            print("  Diffusion модель не найдена, начинаем обучение...")
            diffusion_trainer = train_diffusion_model()

        # Chae модель
        n_measurements = 1275
        input_dim = n_measurements * 4
        output_dim = image_shape[0] * image_shape[1]
        chae_model = ChaeSingleLayerNN(input_dim, output_dim, hidden_dim=1024)
        chae_path = './DATA/models/chae_best.pth'
        if os.path.exists(chae_path):
            chae_model.load_state_dict(torch.load(chae_path, map_location='cpu'), strict=False)
            print("  Chae модель загружена")
        else:
            print("  Chae модель не найдена, начинаем обучение...")
            chae_model = train_chae_model(system_matrix, image_shape)

        # Shang модель
        shang_model = ShangCNN(input_channels=1, output_channels=1, base_filters=32)
        shang_path = './DATA/models/shang_best.pth'
        if os.path.exists(shang_path):
            shang_model.load_state_dict(torch.load(shang_path, map_location='cpu'), strict=False)
            print("  Shang модель загружена")
        else:
            print("  Shang модель не найдена, начинаем обучение...")
            shang_model = train_shang_model()

        # DIP модель (не требует обучения, только создание)
        dip_model = DeepImagePrior(image_shape, latent_dim=100, n_channels=64)
        print("  DIP модель создана")

        # PGNet модель
        pgnet_model = PGNet(input_dim, image_shape, hidden_dim=256, num_heads=8)
        pgnet_path = './DATA/models/pgnet_best.pth'
        if os.path.exists(pgnet_path):
            pgnet_model.load_state_dict(torch.load(pgnet_path, map_location='cpu'), strict=False)
            print("  PGNet модель загружена")
        else:
            print("  PGNet модель не найдена, начинаем обучение...")
            pgnet_model = train_pgnet_model(system_matrix, image_shape)

        # DEQ-MPI модель
        deq_model = DEQMPI(system_matrix, image_shape, n_iterations=3, lambda_param=0.1)
        deq_path = './DATA/models/deq_best.pth'
        if os.path.exists(deq_path):
            checkpoint = torch.load(deq_path, map_location='cpu')
            deq_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("  DEQ-MPI модель загружена")
        else:
            print("  DEQ-MPI модель не найдена, начинаем обучение...")
            deq_model = train_deq_model(system_matrix, image_shape)

    # Сравнение методов
    results = run_full_comparison(
        cnn_trainer, modl_trainer, diffusion_trainer,
        chae_model, shang_model, dip_model, pgnet_model, deq_model
    )

    print("\n" + "=" * 70)
    print("ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН")
    print("=" * 70)
    print("\nРезультаты сохранены в:")
    print("  ./DATA/results/ - изображения сравнения")
    print("  ./DATA/results/all_methods_summary.txt - текстовый отчет")
    print("  ./DATA/results/training_curves/ - кривые обучения")
    print("  ./DATA/models/ - обученные модели")

    return results


def main():
    """Главная функция"""
    import argparse

    parser = argparse.ArgumentParser(description='MPI Reconstruction Pipeline')
    parser.add_argument('--num_samples', type=int, default=5000,
                       help='Number of samples for dataset generation')
    parser.add_argument('--train_models', action='store_true', default=True,
                       help='Train models from scratch')
    parser.add_argument('--load_models', action='store_true',
                       help='Load pre-trained models')

    args = parser.parse_args()

    train_models = args.train_models
    if args.load_models:
        train_models = False

    results = run_pipeline(num_samples=args.num_samples, train_models=train_models)

    if results:
        print("\n" + "="*70)
        print("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ ПО ВСЕМ МЕТОДАМ:")
        print("="*70)

        # Собираем статистику по лучшим методам
        best_ssim = None
        best_method = None

        for result in results:
            distance = result['distance']
            print(f"\n📍 Расстояние: {distance:.3f}")
            for name, data in result['results'].items():
                m = data['metrics']
                print(f"  {name:15s}: SSIM={m['ssim']:.4f}, PSNR={m['psnr']:.2f}дБ, "
                      f"FWHM={m['fwhm']:.2f}px, время={m['time']:.4f}c")

                # Поиск лучшего метода
                if best_ssim is None or m['ssim'] > best_ssim:
                    best_ssim = m['ssim']
                    best_method = name

        print("\n" + "="*70)
        print(f"🏆 ЛУЧШИЙ МЕТОД: {best_method} (SSIM = {best_ssim:.4f})")
        print("="*70)


if __name__ == '__main__':
    main()