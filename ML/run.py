#!/usr/bin/env python3
"""Точка входа для запуска сравнения методов реконструкции MPI"""

import sys
import os

# Добавляем путь к src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main import run_pipeline


if __name__ == '__main__':
    # Параметры запуска
    NUM_SAMPLES = 2000      # Количество образцов для генерации
    TRAIN_MODEL = False     # True - обучать новую модель, False - загрузить существующую

    # Запуск пайплайна
    results = run_pipeline(num_samples=NUM_SAMPLES, train_model=TRAIN_MODEL)

    # Краткий вывод итогов
    if results:
        print("\n" + "=" * 60)
        print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
        print("=" * 60)

        for result in results:
            radius = result['radius']
            distance = result['distance']
            tik_metrics = result['results']['metrics']['tikhonov']
            cnn_metrics = result['results']['metrics']['cnn']

            print(f"\nРадиус: {radius}, Расстояние: {distance}")
            print(f"  SSIM: {tik_metrics['ssim']:.4f} -> {cnn_metrics['ssim']:.4f} "
                  f"(+{cnn_metrics['ssim'] - tik_metrics['ssim']:.4f})")
            print(f"  PSNR: {tik_metrics['psnr']:.2f} -> {cnn_metrics['psnr']:.2f} дБ "
                  f"(+{cnn_metrics['psnr'] - tik_metrics['psnr']:.2f} дБ)")