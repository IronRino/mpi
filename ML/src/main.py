"""Главный модуль для запуска пайплайна с поддержкой всех методов"""

import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from .dataset_generator import MPIDatasetGenerator, MPIDataset
from .trainer import MPITrainer, ModelTrainerFactory
from .comparator import MPIReconstructionComparator
from .models import (ChaeSingleLayerNN, DeepImagePrior, ShangCNN, PGNet, DEQMPI)


def create_directories():
    """Создание необходимых директорий"""
    os.makedirs('./DATA/dataset', exist_ok=True)
    os.makedirs('./DATA/models', exist_ok=True)
    os.makedirs('./DATA/results', exist_ok=True)


def generate_dataset(num_samples=2000):
    """Генерация датасета"""
    print("\n1. Генерация датасета для обучения CNN...")
    generator = MPIDatasetGenerator()
    generator.create_dataset(num_samples=num_samples)
    return generator


def train_cnn_model():
    """Обучение CNN модели"""
    print("\n2. Обучение CNN модели...")

    train_dataset = MPIDataset('./DATA/dataset/X_train.npy', './DATA/dataset/y_train.npy')
    test_dataset = MPIDataset('./DATA/dataset/X_test.npy', './DATA/dataset/y_test.npy')

    sample_measurement, sample_target = train_dataset[0]
    print(f"Размер входных данных: {sample_measurement.shape}")
    print(f"Размер выходных данных: {sample_target.shape}")

    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    trainer = ModelTrainerFactory.create_cnn_trainer(learning_rate=1e-3)

    train_losses, val_losses = trainer.train(
        train_loader, val_loader,
        epochs=30,
        save_path='./DATA/models/cnn_best.pth'
    )

    from .visualization import Visualization
    Visualization.plot_training_curves(
        train_losses, val_losses,
        save_path='./DATA/results/cnn_training_curve.png'
    )

    return trainer


def train_modl_model(system_matrix, image_shape):
    """Обучение MoDL модели"""
    print("\n2b. Обучение MoDL модели...")

    X_train = np.load('./DATA/dataset/X_train.npy')
    y_train = np.load('./DATA/dataset/y_train.npy')

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    class MoDLDataset:
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
            return torch.tensor(meas_vector, dtype=torch.float32), torch.tensor(target, dtype=torch.float32).unsqueeze(0)

    split_idx = int(0.8 * len(X_train))
    train_dataset = MoDLDataset(X_train[:split_idx], y_train[:split_idx])
    val_dataset = MoDLDataset(X_train[split_idx:], y_train[split_idx:])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    trainer = ModelTrainerFactory.create_modl_trainer(
        system_matrix=system_matrix,
        image_shape=image_shape,
        n_iterations=3,
        lambda_param=0.01,
        learning_rate=1e-3
    )

    train_losses, val_losses = trainer.train(
        train_loader, val_loader,
        epochs=20,
        save_path='./DATA/models/modl_best.pth'
    )

    return trainer


def train_chae_model(system_matrix, image_shape):
    """Обучение модели Chae (2017) - однослойная полносвязная сеть"""
    print("\n2c. Обучение модели Chae (2017)...")

    X_train = np.load('./DATA/dataset/X_train.npy')
    y_train = np.load('./DATA/dataset/y_train.npy')

    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")

    # Подготовка данных
    # X_train: (n_samples, 2, n_measurements) где n_measurements = 1275
    # Нужно преобразовать в вектор: real1, imag1, real2, imag2
    n_measurements = X_train.shape[2]  # 1275
    input_dim = n_measurements * 4  # 5100
    output_dim = image_shape[0] * image_shape[1]  # 2601

    print(f"  input_dim = {n_measurements} * 4 = {input_dim}")
    print(f"  output_dim = {image_shape[0]} * {image_shape[1]} = {output_dim}")

    X_train_vec = []
    for meas in X_train:
        # meas имеет форму (2, 1275) - комплексные числа
        real_part = meas.real  # (2, 1275)
        imag_part = meas.imag  # (2, 1275)
        # Объединяем: real1, real2, imag1, imag2 -> (4 * 1275 = 5100)
        meas_vector = np.concatenate([real_part[0], real_part[1], imag_part[0], imag_part[1]])
        X_train_vec.append(meas_vector)

    X_train_vec = np.array(X_train_vec, dtype=np.float32)  # (n_samples, 5100)
    y_train_flat = y_train.reshape(y_train.shape[0], -1)  # (n_samples, 2601)

    print(f"  X_train_vec shape: {X_train_vec.shape}")
    print(f"  y_train_flat shape: {y_train_flat.shape}")

    # Создание модели
    model = ChaeSingleLayerNN(input_dim, output_dim, hidden_dim=512)

    # Используем только часть данных для ускорения (опционально)
    n_samples = min(5000, len(X_train_vec))
    indices = np.random.choice(len(X_train_vec), n_samples, replace=False)
    X_train_vec = X_train_vec[indices]
    y_train_flat = y_train_flat[indices]

    # Подготовка DataLoader
    dataset = TensorDataset(
        torch.tensor(X_train_vec, dtype=torch.float32),
        torch.tensor(y_train_flat, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Обучение
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print(f"  Начало обучения Chae модели на {device}...")
    model.train()
    best_loss = float('inf')

    for epoch in range(50):
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
        scheduler.step(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/50, Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), './DATA/models/chae_best.pth')

    print(f"  Chae модель обучена, лучшая loss: {best_loss:.6f}")

    return model


def train_shang_model():
    """Обучение модели Shang et al. (2020)"""
    print("\n2d. Обучение модели Shang (2020)...")

    import time
    from tqdm import tqdm  # Установите: pip install tqdm

    X_train = np.load('./DATA/dataset/X_train.npy')
    y_train = np.load('./DATA/dataset/y_train.npy')

    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")

    # Используем МАЛЕНЬКУЮ выборку для быстрого обучения
    # Для Shang CNN нужно много памяти, используем только 500 образцов
    n_samples = min(500, len(X_train))
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    X_train = X_train[indices]
    y_train = y_train[indices]

    print(f"  Используется {n_samples} образцов для обучения")

    # Упрощенная подготовка данных - используем случайные начальные реконструкции
    # вместо медленного псевдообращения
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Используется устройство: {device}")

    # Создаем модель
    model = ShangCNN(input_channels=1, output_channels=1, base_filters=32)  # Уменьшили filters
    model = model.to(device)

    # Быстрая подготовка данных - используем случайные или очень простые начальные реконструкции
    X_train_images = []

    print("  Подготовка данных (упрощенная)...")
    for i in tqdm(range(len(X_train)), desc="  Подготовка"):
        meas = X_train[i]

        # Быстрая и простая начальная реконструкция через обратную проекцию
        # (гораздо быстрее, чем псевдообращение)
        real_part = meas.real
        imag_part = meas.imag

        # Простая обратная проекция (adjoint)
        # Приблизительная реконструкция: просто сумма проекций
        initial_recon = np.abs(real_part[0]) + np.abs(real_part[1]) + np.abs(imag_part[0]) + np.abs(imag_part[1])

        # Изменяем размер до 51x51
        from scipy import ndimage
        target_size = 51
        current_size = int(np.sqrt(initial_recon.shape[0]))
        if current_size * current_size != initial_recon.shape[0]:
            # Если не квадрат, делаем простую интерполяцию
            initial_recon = ndimage.zoom(initial_recon.reshape(-1, 1),
                                         (target_size * target_size / len(initial_recon), 1),
                                         order=0).flatten()

        # Изменяем размер до нужного
        if len(initial_recon) != target_size * target_size:
            initial_recon = np.resize(initial_recon, (target_size, target_size))
        else:
            initial_recon = initial_recon.reshape(target_size, target_size)

        # Нормализация
        if initial_recon.max() > 0:
            initial_recon = initial_recon / initial_recon.max()

        X_train_images.append(initial_recon)

    X_train_images = np.array(X_train_images)[:, np.newaxis, :, :].astype(np.float32)
    y_train_images = y_train[:, np.newaxis, :, :].astype(np.float32)

    print(f"  X_train_images shape: {X_train_images.shape}")
    print(f"  y_train_images shape: {y_train_images.shape}")

    # DataLoader
    dataset = TensorDataset(
        torch.tensor(X_train_images, dtype=torch.float32),
        torch.tensor(y_train_images, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)  # num_workers=0 для Windows

    # Обучение
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    print("  Начало обучения Shang модели...")
    model.train()
    best_loss = float('inf')

    for epoch in range(30):  # Уменьшили до 30 эпох
        start_time = time.time()
        total_loss = 0

        # Используем tqdm для отображения прогресса
        pbar = tqdm(loader, desc=f"  Epoch {epoch + 1}/30")
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)
        epoch_time = time.time() - start_time

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/30, Loss: {avg_loss:.6f}, Time: {epoch_time:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), './DATA/models/shang_best.pth')

    print(f"  Shang модель обучена, лучшая loss: {best_loss:.6f}")

    return model


def train_pgnet_model(system_matrix, image_shape):
    """Обучение модели PGNet (Wu et al., 2023)"""
    print("\n2e. Обучение модели PGNet (2023)...")

    import time
    from tqdm import tqdm

    X_train = np.load('./DATA/dataset/X_train.npy')
    y_train = np.load('./DATA/dataset/y_train.npy')

    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")

    # Правильный расчет размеров
    n_measurements = X_train.shape[2]  # 1275
    input_dim = n_measurements * 4  # 5100
    output_dim = image_shape[0] * image_shape[1]  # 2601
    output_shape = image_shape  # (51, 51)

    print(f"  input_dim = {n_measurements} * 4 = {input_dim}")
    print(f"  output_dim = {output_dim}")

    # Используем МАЛЕНЬКУЮ выборку для PGNet (она очень тяжелая)
    n_samples = min(200, len(X_train))  # Всего 200 образцов
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    X_train = X_train[indices]
    y_train = y_train[indices]

    print(f"  Используется {n_samples} образцов для обучения")

    # Подготовка данных
    print("  Подготовка данных...")
    X_train_vec = []
    for i in tqdm(range(len(X_train)), desc="  Преобразование"):
        meas = X_train[i]
        real_part = meas.real  # (2, 1275)
        imag_part = meas.imag  # (2, 1275)
        # Объединяем: real1, real2, imag1, imag2 -> (5100,)
        meas_vector = np.concatenate([real_part[0], real_part[1], imag_part[0], imag_part[1]])
        X_train_vec.append(meas_vector)

    X_train_vec = np.array(X_train_vec, dtype=np.float32)  # (n_samples, 5100)
    y_train_flat = y_train.reshape(y_train.shape[0], -1)  # (n_samples, 2601)

    print(f"  X_train_vec shape: {X_train_vec.shape}")
    print(f"  y_train_flat shape: {y_train_flat.shape}")

    # Создание модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Используется устройство: {device}")

    model = PGNet(input_dim, output_shape, hidden_dim=256)  # Уменьшил hidden_dim с 512 до 256
    model = model.to(device)

    # DataLoader
    dataset = TensorDataset(
        torch.tensor(X_train_vec, dtype=torch.float32),
        torch.tensor(y_train_flat, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True)  # Уменьшил batch_size

    # Обучение
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    print("  Начало обучения PGNet модели...")
    model.train()
    best_loss = float('inf')

    for epoch in range(30):  # Уменьшил до 30 эпох
        start_time = time.time()
        total_loss = 0

        pbar = tqdm(loader, desc=f"  Epoch {epoch + 1}/30")
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            y_pred_flat = y_pred.view(y_pred.shape[0], -1)
            loss = criterion(y_pred_flat, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)
        epoch_time = time.time() - start_time

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/30, Loss: {avg_loss:.6f}, Time: {epoch_time:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), './DATA/models/pgnet_best.pth')

    print(f"  PGNet модель обучена, лучшая loss: {best_loss:.6f}")

    return model


def train_deq_model(system_matrix, image_shape):
    """Обучение модели DEQ-MPI (Güngör et al., 2024)"""
    print("\n2f. Обучение модели DEQ-MPI (2024)...")

    import time
    from tqdm import tqdm

    X_train = np.load('./DATA/dataset/X_train.npy')
    y_train = np.load('./DATA/dataset/y_train.npy')

    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")

    # Используем ОЧЕНЬ МАЛЕНЬКУЮ выборку для DEQ (она очень тяжелая из-за итераций)
    n_samples = min(100, len(X_train))  # Всего 100 образцов для начала
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    X_train = X_train[indices]
    y_train = y_train[indices]

    print(f"  Используется {n_samples} образцов для обучения")

    # Подготовка данных
    n_measurements = X_train.shape[2]  # 1275
    input_dim = n_measurements * 4  # 5100

    print(f"  input_dim = {input_dim}")

    X_train_vec = []
    print("  Подготовка данных...")
    for i in tqdm(range(len(X_train)), desc="  Преобразование"):
        meas = X_train[i]
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

    # Уменьшаем количество итераций для ускорения
    n_iterations = 3  # Уменьшил с 5 до 3
    lambda_param = 0.1

    # Создание модели
    model = DEQMPI(system_matrix, image_shape, n_iterations=n_iterations, lambda_param=lambda_param)
    model = model.to(device)

    # Подготовка DataLoader с маленьким batch_size
    dataset = TensorDataset(
        torch.tensor(X_train_vec, dtype=torch.float32),
        torch.tensor(y_train_img[:, np.newaxis, :, :], dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)  # batch_size=4

    # Обучение
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Уменьшил learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    print(f"  Начало обучения DEQ-MPI модели (итераций={n_iterations})...")
    print("  ВНИМАНИЕ: DEQ-MPI обучается медленно из-за итеративной природы!")

    model.train()
    best_loss = float('inf')
    epochs = 20  # Уменьшил до 20 эпох

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0

        # Используем tqdm для отображения прогресса
        pbar = tqdm(loader, desc=f"  Epoch {epoch + 1}/{epochs}", leave=True)

        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            # Forward pass
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            # Backward pass
            loss.backward()

            # Обрезка градиентов для стабильности
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

            # Обновляем прогресс-бар
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg_loss': f'{total_loss / len(loader):.6f}'
            })

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)

        epoch_time = time.time() - start_time

        # Вывод каждые 5 эпох
        if (epoch + 1) % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}, "
                  f"LR: {current_lr:.2e}, Time: {epoch_time:.1f}s")

        # Сохраняем лучшую модель
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
            print(f"  * Сохранена лучшая модель (loss: {best_loss:.6f})")

    print(f"  DEQ-MPI модель обучена, лучшая loss: {best_loss:.6f}")

    return model


def train_diffusion_model():
    """Обучение диффузионной модели"""
    print("\n2g. Обучение Diffusion модели...")

    train_dataset = MPIDataset('./DATA/dataset/X_train.npy', './DATA/dataset/y_train.npy')
    test_dataset = MPIDataset('./DATA/dataset/X_test.npy', './DATA/dataset/y_test.npy')

    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    trainer = ModelTrainerFactory.create_diffusion_trainer(
        n_steps=500,
        learning_rate=1e-4,
        image_size=51  # Добавляем размер изображения
    )

    train_losses, val_losses = trainer.train(
        train_loader, val_loader,
        epochs=30,
        save_path='./DATA/models/diffusion_best.pth'
    )

    # Визуализация кривых обучения
    from .visualization import Visualization
    Visualization.plot_training_curves(
        train_losses, val_losses,
        save_path='./DATA/results/diffusion_training_curve.png'
    )

    return trainer


def train_hybrid_model(system_matrix, image_shape):
    """Обучение гибридной модели"""
    print("\n2h. Обучение Hybrid модели...")

    X_train = np.load('./DATA/dataset/X_train.npy')
    y_train = np.load('./DATA/dataset/y_train.npy')

    class HybridDataset:
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
            return torch.tensor(meas_vector, dtype=torch.float32), torch.tensor(target, dtype=torch.float32).unsqueeze(0)

    split_idx = int(0.8 * len(X_train))
    train_dataset = HybridDataset(X_train[:split_idx], y_train[:split_idx])
    val_dataset = HybridDataset(X_train[split_idx:], y_train[split_idx:])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    trainer = ModelTrainerFactory.create_hybrid_trainer(
        system_matrix=system_matrix,
        image_shape=image_shape,
        modl_iterations=3,
        diffusion_steps=50,
        learning_rate=1e-4
    )

    train_losses, val_losses = trainer.train(
        train_loader, val_loader,
        epochs=20,
        save_path='./DATA/models/hybrid_best.pth'
    )

    return trainer


def load_cnn_model():
    """Загрузка сохраненной CNN модели"""
    print("\nЗагрузка обученной CNN модели...")
    if not os.path.exists('./DATA/models/cnn_best.pth'):
        print("  Файл модели не найден, будет обучена новая модель")
        return train_cnn_model()

    trainer = ModelTrainerFactory.create_cnn_trainer()
    checkpoint = trainer.load_model('./DATA/models/cnn_best.pth')
    return trainer


def load_modl_model(system_matrix, image_shape):
    """Загрузка сохраненной MoDL модели"""
    print("\nЗагрузка обученной MoDL модели...")
    if not os.path.exists('./DATA/models/modl_best.pth'):
        print("  Файл модели не найден, будет обучена новая модель")
        return train_modl_model(system_matrix, image_shape)

    trainer = ModelTrainerFactory.create_modl_trainer(system_matrix, image_shape)
    trainer.load_model('./DATA/models/modl_best.pth')
    return trainer


def load_chae_model(system_matrix, image_shape):
    """Загрузка модели Chae (2017)"""
    print("\nЗагрузка модели Chae (2017)...")

    # Загружаем данные для определения правильных размеров
    X_train = np.load('./DATA/dataset/X_train.npy')
    y_train = np.load('./DATA/dataset/y_train.npy')

    # Правильный расчет размеров
    # X_train имеет форму (n_samples, 2, n_measurements), где n_measurements = 1275
    # Для Chae нужно объединить real и imag части для 2 катушек
    # Вход: real1, imag1, real2, imag2 = 4 * n_measurements = 4 * 1275 = 5100
    n_measurements = X_train.shape[2]  # 1275
    input_dim = n_measurements * 4  # 1275 * 4 = 5100
    output_dim = image_shape[0] * image_shape[1]  # 51 * 51 = 2601

    print(f"  Chae модель: input_dim={input_dim}, output_dim={output_dim}")

    model = ChaeSingleLayerNN(input_dim, output_dim, hidden_dim=512)

    model_path = './DATA/models/chae_best.pth'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        # Проверяем соответствие размеров перед загрузкой
        if 'network.0.weight' in checkpoint:
            saved_shape = checkpoint['network.0.weight'].shape
            current_shape = model.network[0].weight.shape
            if saved_shape != current_shape:
                print(f"  ВНИМАНИЕ: Несоответствие размеров модели!")
                print(f"    Сохраненная: {saved_shape}")
                print(f"    Текущая: {current_shape}")
                print(f"  Будет обучена новая модель")
                model = train_chae_model(system_matrix, image_shape)
            else:
                model.load_state_dict(checkpoint)
                print("  Модель Chae загружена из файла")
        else:
            print("  Неверный формат сохраненной модели, обучение новой...")
            model = train_chae_model(system_matrix, image_shape)
    else:
        print("  Файл модели не найден, будет обучена новая модель")
        model = train_chae_model(system_matrix, image_shape)

    model.eval()
    return model


def load_shang_model():
    """Загрузка модели Shang et al. (2020)"""
    print("\nЗагрузка модели Shang (2020)...")

    model_path = './DATA/models/shang_best.pth'

    if os.path.exists(model_path):
        # Загружаем checkpoint для анализа
        checkpoint = torch.load(model_path, map_location='cpu')

        # Определяем параметры из сохраненной модели
        # Проверяем размер первого сверточного слоя
        if 'conv1.0.weight' in checkpoint:
            saved_filters = checkpoint['conv1.0.weight'].shape[0]
            print(f"  Обнаружена модель с base_filters={saved_filters}")

            # Создаем модель с правильными параметрами
            model = ShangCNN(input_channels=1, output_channels=1, base_filters=saved_filters)
            model.load_state_dict(checkpoint)
            print("  Модель Shang загружена из файла")
        else:
            print("  Неверный формат сохраненной модели")
            model = ShangCNN(input_channels=1, output_channels=1, base_filters=32)
            print("  Создана новая модель с base_filters=32")
    else:
        print("  Файл модели не найден, создана новая модель")
        model = ShangCNN(input_channels=1, output_channels=1, base_filters=32)

    model.eval()
    return model


def load_pgnet_model(system_matrix, image_shape):
    """Загрузка модели PGNet (Wu et al., 2023)"""
    print("\nЗагрузка модели PGNet (2023)...")

    # Загружаем данные для определения правильных размеров
    X_train = np.load('./DATA/dataset/X_train.npy')

    # Правильный расчет размеров
    n_measurements = X_train.shape[2]  # 1275
    input_dim = n_measurements * 4  # 1275 * 4 = 5100
    output_shape = image_shape  # (51, 51)

    print(f"  PGNet: input_dim={input_dim}, output_shape={output_shape}")

    model_path = './DATA/models/pgnet_best.pth'

    if os.path.exists(model_path):
        # Загружаем checkpoint для анализа
        checkpoint = torch.load(model_path, map_location='cpu')

        # Проверяем размеры
        if 'projection_encoder.0.weight' in checkpoint:
            saved_input_dim = checkpoint['projection_encoder.0.weight'].shape[1]
            print(f"  Сохраненная модель: input_dim={saved_input_dim}")
            print(f"  Требуемая модель: input_dim={input_dim}")

            if saved_input_dim != input_dim:
                print(f"  НЕСООТВЕТСТВИЕ РАЗМЕРОВ! Удаляем старый файл...")
                os.remove(model_path)
                print("  Будет создана новая модель. Для обучения установите TRAIN_MODELS=True")
                model = PGNet(input_dim, output_shape, hidden_dim=512)
            else:
                model = PGNet(input_dim, output_shape, hidden_dim=512)
                model.load_state_dict(checkpoint)
                print("  Модель PGNet загружена из файла")
        else:
            print("  Неверный формат сохраненной модели")
            model = PGNet(input_dim, output_shape, hidden_dim=512)
    else:
        print("  Файл модели не найден")
        model = PGNet(input_dim, output_shape, hidden_dim=512)
        print("  Создана новая модель. Для обучения установите TRAIN_MODELS=True")

    model.eval()
    return model


def load_deq_model(system_matrix, image_shape):
    """Загрузка модели DEQ-MPI (Güngör et al., 2024)"""
    print("\nЗагрузка модели DEQ-MPI (2024)...")

    model_path = './DATA/models/deq_best.pth'

    # Параметры по умолчанию
    n_iterations = 3
    lambda_param = 0.1

    if os.path.exists(model_path):
        try:
            # Загружаем checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')

            # Извлекаем параметры из сохраненной модели
            if 'n_iterations' in checkpoint:
                n_iterations = checkpoint['n_iterations']
            if 'lambda_param' in checkpoint:
                lambda_param = checkpoint['lambda_param']

            print(f"  Загружаются параметры: n_iterations={n_iterations}, lambda_param={lambda_param}")

            # Создаем модель с правильными параметрами
            model = DEQMPI(system_matrix, image_shape, n_iterations=n_iterations, lambda_param=lambda_param)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("  Модель DEQ-MPI загружена из файла")

        except Exception as e:
            print(f"  Ошибка загрузки модели: {e}")
            print("  Удаляем старый файл и создаем новую модель...")
            os.remove(model_path)
            model = DEQMPI(system_matrix, image_shape, n_iterations=n_iterations, lambda_param=lambda_param)
    else:
        print("  Файл модели не найден, создана новая модель")
        model = DEQMPI(system_matrix, image_shape, n_iterations=n_iterations, lambda_param=lambda_param)
        print("  Для обучения установите TRAIN_MODELS=True")

    model.eval()
    return model


def load_diffusion_model():
    """Загрузка сохраненной диффузионной модели"""
    print("\nЗагрузка обученной Diffusion модели...")
    if not os.path.exists('./DATA/models/diffusion_best.pth'):
        print("  Файл модели не найден, будет обучена новая модель")
        return train_diffusion_model()
    trainer = ModelTrainerFactory.create_diffusion_trainer()
    trainer.load_model('./DATA/models/diffusion_best.pth')
    return trainer


def load_hybrid_model(system_matrix, image_shape):
    """Загрузка сохраненной гибридной модели"""
    print("\nЗагрузка обученной Hybrid модели...")
    if not os.path.exists('./DATA/models/hybrid_best.pth'):
        print("  Файл модели не найден, будет обучена новая модель")
        return train_hybrid_model(system_matrix, image_shape)

    trainer = ModelTrainerFactory.create_hybrid_trainer(system_matrix, image_shape)
    trainer.load_model('./DATA/models/hybrid_best.pth')
    return trainer


def run_full_comparison(cnn_trainer, modl_trainer, diffusion_trainer, #hybrid_trainer,
                        chae_model, shang_model, pgnet_model, deq_model):
    """Запуск полного сравнения всех методов"""
    print("\n3. Сравнение всех методов реконструкции...")

    comparator = MPIReconstructionComparator('./../Chinese data/SystemMatrix.h5')
    comparator.set_cnn_model(cnn_trainer)
    comparator.set_modl_model(modl_trainer)
    comparator.set_diffusion_model(diffusion_trainer)
    #comparator.set_hybrid_model(hybrid_trainer)

    # Установка новых моделей
    comparator.set_chae_model(chae_model)
    comparator.set_shang_model(shang_model)
    comparator.set_pgnet_model(pgnet_model)
    comparator.set_deq_model(deq_model)

    radius = 0.2
    distances = [0.2, 0.15, 0.1, 0.05]

    print(f"\nТестовые случаи:")
    print(f"  Радиус капель: {radius}")
    print(f"  Расстояния между центрами: {distances}")

    results = comparator.run_full_comparison(radius=radius, distances=distances)

    comparator.print_summary_table()
    comparator.save_results_to_file()

    return results


def run_pipeline(num_samples=2000, train_models=True):
    """Запуск полного пайплайна с обучением всех моделей"""
    print("=" * 70)
    print("ПАЙПЛАЙН СРАВНЕНИЯ МЕТОДОВ РЕКОНСТРУКЦИИ MPI")
    print("=" * 70)
    print("\nДоступные методы:")
    print("  - Тихонов (классический)")
    print("  - KatsMarc (Kaczmarz, 1937)")
    print("  - Chae (2017) - однослойная полносвязная сеть")
    print("  - DIP (Dittmer et al., 2020) - Deep Image Prior")
    print("  - Shang (2020) - CNN для улучшения разрешения")
    print("  - PGNet (Wu et al., 2023) - Projection Generation Network")
    print("  - DEQ-MPI (Güngör et al., 2024)")
    print("  - CNN (UNet)")
    print("  - MoDL")
    print("  - Diffusion")
    print("  - Hybrid (MoDL+Diffusion)")

    create_directories()

    # Генерация датасета
    if not os.path.exists('./DATA/dataset/X_train.npy'):
        generator = generate_dataset(num_samples)
        system_matrix = generator.SM
        image_shape = generator.image_shape
    else:
        print("\n1. Датасет уже существует, пропускаем генерацию...")
        generator = MPIDatasetGenerator()
        system_matrix = generator.SM
        image_shape = generator.image_shape

    print(f"\nСистемная матрица: {system_matrix.shape}")
    print(f"Размер изображения: {image_shape}")

    # Обучение или загрузка моделей
    if train_models:
        print("\n--- Режим: Обучение новых моделей ---")
        cnn_trainer = train_cnn_model()
        modl_trainer = train_modl_model(system_matrix, image_shape)
        diffusion_trainer = train_diffusion_model()
        hybrid_trainer = train_hybrid_model(system_matrix, image_shape)
        chae_model = train_chae_model(system_matrix, image_shape)
        shang_model = train_shang_model()
        pgnet_model = train_pgnet_model(system_matrix, image_shape)
        deq_model = train_deq_model(system_matrix, image_shape)
    else:
        print("\n--- Режим: Загрузка сохраненных моделей ---")
        cnn_trainer = load_cnn_model()
        modl_trainer = load_modl_model(system_matrix, image_shape)
        diffusion_trainer = load_diffusion_model()
        #hybrid_trainer = load_hybrid_model(system_matrix, image_shape)
        chae_model = load_chae_model(system_matrix, image_shape)
        shang_model = load_shang_model()
        pgnet_model = load_pgnet_model(system_matrix, image_shape)
        deq_model = load_deq_model(system_matrix, image_shape)

    # Сравнение методов
    results = run_full_comparison(cnn_trainer, modl_trainer, diffusion_trainer, #hybrid_trainer,
                                  chae_model, shang_model, pgnet_model, deq_model)

    print("\n" + "=" * 70)
    print("ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН!")
    print("=" * 70)

    return results


if __name__ == '__main__':
    NUM_SAMPLES = 2000
    TRAIN_MODELS = False

    results = run_pipeline(num_samples=NUM_SAMPLES, train_models=TRAIN_MODELS)

    if results:
        print("\n" + "=" * 70)
        print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
        print("=" * 70)

        for result in results:
            distance = result['distance']
            print(f"\nРасстояние: {distance:.3f}")
            for name, data in result['results'].items():
                m = data['metrics']
                print(f"  {name}: SSIM={m['ssim']:.4f}, PSNR={m['psnr']:.2f}дБ, FWHM={m['fwhm']:.2f}px, время={m['time']:.4f}c")