"""Главный модуль для запуска пайплайна с поддержкой всех методов"""

import os
import torch
from torch.utils.data import DataLoader
import numpy as np

from .dataset_generator import MPIDatasetGenerator, MPIDataset
from .trainer import MPITrainer, ModelTrainerFactory
from .comparator import MPIReconstructionComparator


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

    # Создаем датасет для MoDL с разделением на реальную и мнимую части
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


def train_diffusion_model():
    """Обучение диффузионной модели"""
    print("\n2c. Обучение Diffusion модели...")

    train_dataset = MPIDataset('./DATA/dataset/X_train.npy', './DATA/dataset/y_train.npy')
    test_dataset = MPIDataset('./DATA/dataset/X_test.npy', './DATA/dataset/y_test.npy')

    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    trainer = ModelTrainerFactory.create_diffusion_trainer(
        n_steps=500,
        learning_rate=1e-4
    )

    train_losses, val_losses = trainer.train(
        train_loader, val_loader,
        epochs=30,
        save_path='./DATA/models/diffusion_best.pth'
    )

    return trainer


def train_hybrid_model(system_matrix, image_shape):
    """Обучение гибридной модели"""
    print("\n2d. Обучение Hybrid модели...")

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
    print("\n2. Загрузка обученной CNN модели...")
    if not os.path.exists('./DATA/models/cnn_best.pth'):
        print("  Файл модели не найден, будет обучена новая модель")
        return train_cnn_model()

    trainer = ModelTrainerFactory.create_cnn_trainer()
    checkpoint = trainer.load_model('./DATA/models/cnn_best.pth')
    return trainer


def load_modl_model(system_matrix, image_shape):
    """Загрузка сохраненной MoDL модели"""
    print("\n2b. Загрузка обученной MoDL модели...")
    if not os.path.exists('./DATA/models/modl_best.pth'):
        print("  Файл модели не найден, будет обучена новая модель")
        return train_modl_model(system_matrix, image_shape)

    trainer = ModelTrainerFactory.create_modl_trainer(system_matrix, image_shape)
    trainer.load_model('./DATA/models/modl_best.pth')
    return trainer


def load_diffusion_model():
    """Загрузка сохраненной диффузионной модели"""
    print("\n2c. Загрузка обученной Diffusion модели...")
    if not os.path.exists('./DATA/models/diffusion_best.pth'):
        print("  Файл модели не найден, будет обучена новая модель")
        return train_diffusion_model()
    trainer = ModelTrainerFactory.create_diffusion_trainer()
    trainer.load_model('./DATA/models/diffusion_best.pth')
    return trainer


def load_hybrid_model(system_matrix, image_shape):
    """Загрузка сохраненной гибридной модели"""
    print("\n2d. Загрузка обученной Hybrid модели...")
    if not os.path.exists('./DATA/models/hybrid_best.pth'):
        print("  Файл модели не найден, будет обучена новая модель")
        return train_hybrid_model(system_matrix, image_shape)

    trainer = ModelTrainerFactory.create_hybrid_trainer(system_matrix, image_shape)
    trainer.load_model('./DATA/models/hybrid_best.pth')
    return trainer


def run_full_comparison(cnn_trainer, modl_trainer, diffusion_trainer, hybrid_trainer):
    """Запуск полного сравнения всех методов"""
    print("\n3. Сравнение всех методов реконструкции...")

    comparator = MPIReconstructionComparator('./../Chinese data/SystemMatrix.h5')
    comparator.set_cnn_model(cnn_trainer)
    comparator.set_modl_model(modl_trainer)
    comparator.set_diffusion_model(diffusion_trainer)
    comparator.set_hybrid_model(hybrid_trainer)

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

    # Обучение или загрузка моделей в зависимости от параметра
    if train_models:
        print("\n--- Режим: Обучение новых моделей ---")
        cnn_trainer = train_cnn_model()
        modl_trainer = train_modl_model(system_matrix, image_shape)
        diffusion_trainer = None
        hybrid_trainer = None
    else:
        print("\n--- Режим: Загрузка сохраненных моделей ---")
        cnn_trainer = load_cnn_model()
        modl_trainer = load_modl_model(system_matrix, image_shape)
        diffusion_trainer = None
        hybrid_trainer = None

    # Сравнение методов
    results = run_full_comparison(cnn_trainer, modl_trainer, diffusion_trainer, hybrid_trainer)

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