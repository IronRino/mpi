"""Главный модуль для запуска пайплайна"""

import os
import torch
from torch.utils.data import DataLoader

from .dataset_generator import MPIDatasetGenerator, MPIDataset
from .trainer import MPITrainer
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

    # Загрузка датасета
    train_dataset = MPIDataset('./DATA/dataset/X_train.npy', './DATA/dataset/y_train.npy')
    test_dataset = MPIDataset('./DATA/dataset/X_test.npy', './DATA/dataset/y_test.npy')

    # Проверка размеров
    sample_measurement, sample_target = train_dataset[0]
    print(f"Размер входных данных: {sample_measurement.shape}")
    print(f"Размер выходных данных: {sample_target.shape}")

    # Создание DataLoader
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Создание и обучение модели
    trainer = MPITrainer()
    trainer.create_model(input_channels=4)
    trainer.setup_training(learning_rate=1e-3)

    train_losses, val_losses = trainer.train(
        train_loader, val_loader,
        epochs=30,
        save_path='./DATA/models/mpi_cnn_best.pth'
    )

    # График обучения
    from .visualization import Visualization
    Visualization.plot_training_curves(
        train_losses, val_losses,
        save_path='./DATA/results/cnn_training_curve.png'
    )

    return trainer


def load_cnn_model():
    """Загрузка сохраненной CNN модели"""
    print("\n2. Загрузка обученной CNN модели...")
    trainer = MPITrainer()
    trainer.create_model(input_channels=4)
    trainer.setup_training()

    if torch.cuda.is_available():
        trainer.load_model('./DATA/models/mpi_cnn_best.pth')
    else:
        checkpoint = torch.load('./DATA/models/mpi_cnn_best.pth',
                                map_location=torch.device('cpu'))
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Модель загружена из эпохи {checkpoint['epoch'] + 1}")

    return trainer


def run_comparison(cnn_trainer):
    """Запуск сравнения методов"""
    print("\n3. Сравнение методов реконструкции...")

    comparator = MPIReconstructionComparator()
    comparator.set_cnn_model(cnn_trainer)

    # Запуск экспериментов
    radius = 0.2
    distances = [0.2, 0.1, 0.05]

    print(f"\nТестовые случаи:")
    print(f"  Радиус капель: {radius}")
    print(f"  Расстояния между центрами: {distances}")

    results = comparator.run_comparison_experiments(
        radius_list=[radius],
        distance_list=distances
    )

    # Вывод результатов
    comparator.print_summary()
    comparator.save_results_to_file()
    comparator.plot_summary_results()

    return results


def run_pipeline(num_samples=2000, train_model=True):
    """Запуск полного пайплайна"""
    print("=" * 60)
    print("ПАЙПЛАЙН СРАВНЕНИЯ МЕТОДОВ РЕКОНСТРУКЦИИ MPI")
    print("=" * 60)

    # Создание директорий
    create_directories()

    # Генерация датасета (если не существует)
    if not os.path.exists('./DATA/dataset/X_train.npy'):
        generate_dataset(num_samples)
    else:
        print("\n1. Датасет уже существует, пропускаем генерацию...")

    # Обучение или загрузка модели
    if train_model or not os.path.exists('./DATA/models/mpi_cnn_best.pth'):
        cnn_trainer = train_cnn_model()
    else:
        cnn_trainer = load_cnn_model()

    # Сравнение методов
    results = run_comparison(cnn_trainer)

    print("\n" + "=" * 60)
    print("ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН!")
    print("=" * 60)
    print("\nРезультаты сохранены в:")
    print("  ./DATA/results/ - изображения сравнения")
    print("  ./DATA/results/summary_results.txt - текстовый отчет")
    print("  ./DATA/results/summary_plot.png - сводный график")

    return results