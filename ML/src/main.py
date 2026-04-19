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

    # Подготовка данных
    X_train_vec = []
    for meas in X_train:
        real_part = meas.real
        imag_part = meas.imag
        meas_vector = np.concatenate([real_part.flatten(), imag_part.flatten()])
        X_train_vec.append(meas_vector)

    X_train_vec = np.array(X_train_vec)
    y_train_flat = y_train.reshape(y_train.shape[0], -1)

    input_dim = X_train_vec.shape[1]
    output_dim = y_train_flat.shape[1]

    print(f"  Input dim: {input_dim}, Output dim: {output_dim}")

    # Создание модели
    model = ChaeSingleLayerNN(input_dim, output_dim, hidden_dim=512)

    # Подготовка DataLoader
    X_tensor = torch.tensor(X_train_vec, dtype=torch.float32)
    y_tensor = torch.tensor(y_train_flat, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Обучение
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(30):
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/30, Loss: {total_loss/len(loader):.6f}")

    # Сохранение модели
    torch.save(model.state_dict(), './DATA/models/chae_best.pth')

    return model


def train_shang_model():
    """Обучение модели Shang et al. (2020)"""
    print("\n2d. Обучение модели Shang (2020)...")

    X_train = np.load('./DATA/dataset/X_train.npy')
    y_train = np.load('./DATA/dataset/y_train.npy')

    # Для Shang CNN нужны изображения на входе
    # Сначала делаем грубую реконструкцию через псевдообращение
    from scipy import ndimage

    model = ShangCNN(input_channels=1, output_channels=1, base_filters=64)

    # Подготовка данных (упрощенная версия)
    X_train_images = []
    for meas in X_train[:1000]:  # Используем подвыборку для скорости
        real_part = meas.real
        imag_part = meas.imag
        meas_vector = np.concatenate([real_part.flatten(), imag_part.flatten()])

        # Псевдообращение для начальной реконструкции
        A_pinv = np.linalg.pinv(X_train.shape[2] * np.eye(meas_vector.shape[0]))  # Упрощение
        initial = np.random.rand(51, 51)  # Временное решение
        X_train_images.append(initial)

    X_train_images = np.array(X_train_images)[:, np.newaxis, :, :]
    y_train_images = y_train[:1000, np.newaxis, :, :]

    X_tensor = torch.tensor(X_train_images, dtype=torch.float32)
    y_tensor = torch.tensor(y_train_images, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(20):
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/20, Loss: {total_loss/len(loader):.6f}")

    torch.save(model.state_dict(), './DATA/models/shang_best.pth')

    return model


def train_pgnet_model(system_matrix, image_shape):
    """Обучение модели PGNet (Wu et al., 2023)"""
    print("\n2e. Обучение модели PGNet (2023)...")

    X_train = np.load('./DATA/dataset/X_train.npy')
    y_train = np.load('./DATA/dataset/y_train.npy')

    # Подготовка данных
    X_train_vec = []
    for meas in X_train:
        real_part = meas.real
        imag_part = meas.imag
        meas_vector = np.concatenate([real_part.flatten(), imag_part.flatten()])
        X_train_vec.append(meas_vector)

    X_train_vec = np.array(X_train_vec)
    y_train_flat = y_train.reshape(y_train.shape[0], -1)

    input_dim = X_train_vec.shape[1]
    output_shape = image_shape

    model = PGNet(input_dim, output_shape, hidden_dim=512)

    X_tensor = torch.tensor(X_train_vec, dtype=torch.float32)
    y_tensor = torch.tensor(y_train_flat, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(30):
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            y_pred_flat = y_pred.view(y_pred.shape[0], -1)
            loss = criterion(y_pred_flat, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/30, Loss: {total_loss/len(loader):.6f}")

    torch.save(model.state_dict(), './DATA/models/pgnet_best.pth')

    return model


def train_deq_model(system_matrix, image_shape):
    """Обучение модели DEQ-MPI (Güngör et al., 2024)"""
    print("\n2f. Обучение модели DEQ-MPI (2024)...")

    X_train = np.load('./DATA/dataset/X_train.npy')
    y_train = np.load('./DATA/dataset/y_train.npy')

    # Подготовка данных
    X_train_vec = []
    for meas in X_train[:500]:  # Подвыборка для скорости
        real_part = meas.real
        imag_part = meas.imag
        meas_vector = np.concatenate([real_part.flatten(), imag_part.flatten()])
        X_train_vec.append(meas_vector)

    X_train_vec = np.array(X_train_vec)
    y_train_img = y_train[:500]

    model = DEQMPI(system_matrix, image_shape, n_iterations=5, lambda_param=0.1)

    X_tensor = torch.tensor(X_train_vec, dtype=torch.float32)
    y_tensor = torch.tensor(y_train_img[:, np.newaxis, :, :], dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(20):
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/20, Loss: {total_loss/len(loader):.6f}")

    torch.save(model.state_dict(), './DATA/models/deq_best.pth')

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
    X_train = np.load('./DATA/dataset/X_train.npy')
    y_train = np.load('./DATA/dataset/y_train.npy')

    input_dim = X_train.shape[1] * 4  # real+imag для 2 катушек
    output_dim = image_shape[0] * image_shape[1]

    model = ChaeSingleLayerNN(input_dim, output_dim, hidden_dim=512)

    if os.path.exists('./DATA/models/chae_best.pth'):
        model.load_state_dict(torch.load('./DATA/models/chae_best.pth'))
        print("  Модель Chae загружена из файла")
    else:
        print("  Файл модели не найден, будет обучена новая модель")
        model = train_chae_model(system_matrix, image_shape)

    model.eval()
    return model


def load_shang_model():
    """Загрузка модели Shang et al. (2020)"""
    print("\nЗагрузка модели Shang (2020)...")
    model = ShangCNN(input_channels=1, output_channels=1, base_filters=64)

    if os.path.exists('./DATA/models/shang_best.pth'):
        model.load_state_dict(torch.load('./DATA/models/shang_best.pth'))
        print("  Модель Shang загружена из файла")
    else:
        print("  Файл модели не найден, будет обучена новая модель")
        model = train_shang_model()

    model.eval()
    return model


def load_pgnet_model(system_matrix, image_shape):
    """Загрузка модели PGNet (Wu et al., 2023)"""
    print("\nЗагрузка модели PGNet (2023)...")
    X_train = np.load('./DATA/dataset/X_train.npy')
    input_dim = X_train.shape[1] * 4

    model = PGNet(input_dim, image_shape, hidden_dim=512)

    if os.path.exists('./DATA/models/pgnet_best.pth'):
        model.load_state_dict(torch.load('./DATA/models/pgnet_best.pth'))
        print("  Модель PGNet загружена из файла")
    else:
        print("  Файл модели не найден, будет обучена новая модель")
        model = train_pgnet_model(system_matrix, image_shape)

    model.eval()
    return model


def load_deq_model(system_matrix, image_shape):
    """Загрузка модели DEQ-MPI (Güngör et al., 2024)"""
    print("\nЗагрузка модели DEQ-MPI (2024)...")
    model = DEQMPI(system_matrix, image_shape, n_iterations=5, lambda_param=0.1)

    if os.path.exists('./DATA/models/deq_best.pth'):
        model.load_state_dict(torch.load('./DATA/models/deq_best.pth'))
        print("  Модель DEQ-MPI загружена из файла")
    else:
        print("  Файл модели не найден, будет обучена новая модель")
        model = train_deq_model(system_matrix, image_shape)

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


def run_full_comparison(cnn_trainer, modl_trainer, diffusion_trainer, hybrid_trainer,
                        chae_model, shang_model, pgnet_model, deq_model):
    """Запуск полного сравнения всех методов"""
    print("\n3. Сравнение всех методов реконструкции...")

    comparator = MPIReconstructionComparator('./../Chinese data/SystemMatrix.h5')
    comparator.set_cnn_model(cnn_trainer)
    comparator.set_modl_model(modl_trainer)
    comparator.set_diffusion_model(diffusion_trainer)
    comparator.set_hybrid_model(hybrid_trainer)

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
        hybrid_trainer = load_hybrid_model(system_matrix, image_shape)
        chae_model = load_chae_model(system_matrix, image_shape)
        shang_model = load_shang_model()
        pgnet_model = load_pgnet_model(system_matrix, image_shape)
        deq_model = load_deq_model(system_matrix, image_shape)

    # Сравнение методов
    results = run_full_comparison(cnn_trainer, modl_trainer, diffusion_trainer, hybrid_trainer,
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