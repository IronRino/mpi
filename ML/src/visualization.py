"""Визуализация результатов реконструкции"""

import matplotlib.pyplot as plt
import numpy as np


class Visualization:
    """Класс для визуализации результатов"""

    @staticmethod
    def plot_comparison(original, tikhonov_recon, cnn_recon,
                        radius=None, distance=None,
                        tikhonov_metrics=None, cnn_metrics=None,
                        save_path=None):
        """Визуализация сравнения методов"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        vmin, vmax = 0, 1

        # Оригинал
        im0 = axes[0, 0].imshow(original, cmap='hot', vmin=vmin, vmax=vmax)
        axes[0, 0].set_title('Оригинальное изображение')
        title = ""
        if radius is not None and distance is not None:
            title = f'radius={radius}, distance={distance}'
        axes[0, 0].set_xlabel(title)
        plt.colorbar(im0, ax=axes[0, 0])

        # Тихонов
        im1 = axes[0, 1].imshow(tikhonov_recon, cmap='hot', vmin=vmin, vmax=vmax)
        axes[0, 1].set_title('Метод Тихонова')
        if tikhonov_metrics:
            axes[0, 1].set_xlabel(
                f"SSIM: {tikhonov_metrics['ssim']:.4f}\n"
                f"PSNR: {tikhonov_metrics['psnr']:.2f} дБ\n"
                f"FWHM: {tikhonov_metrics['fwhm']:.2f} пикс."
            )
        plt.colorbar(im1, ax=axes[0, 1])

        # CNN
        im2 = axes[0, 2].imshow(cnn_recon, cmap='hot', vmin=vmin, vmax=vmax)
        axes[0, 2].set_title('CNN метод')
        if cnn_metrics:
            axes[0, 2].set_xlabel(
                f"SSIM: {cnn_metrics['ssim']:.4f}\n"
                f"PSNR: {cnn_metrics['psnr']:.2f} дБ\n"
                f"FWHM: {cnn_metrics['fwhm']:.2f} пикс."
            )
        plt.colorbar(im2, ax=axes[0, 2])

        # Профили
        center_y = original.shape[0] // 2
        x = np.linspace(-1, 1, original.shape[1])

        axes[1, 0].plot(x, original[center_y, :], 'k-', linewidth=2, label='Оригинал')
        axes[1, 0].plot(x, tikhonov_recon[center_y, :], 'r--', linewidth=1.5, label='Тихонов')
        axes[1, 0].plot(x, cnn_recon[center_y, :], 'b:', linewidth=1.5, label='CNN')
        axes[1, 0].set_xlabel('X координата')
        axes[1, 0].set_ylabel('Интенсивность')
        axes[1, 0].set_title('Горизонтальный профиль через центр')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Ошибки
        diff_tikhonov = np.abs(original - tikhonov_recon)
        diff_cnn = np.abs(original - cnn_recon)

        im3 = axes[1, 1].imshow(diff_tikhonov, cmap='Reds', vmin=0, vmax=0.5)
        axes[1, 1].set_title('Ошибка: Тихонов')
        if tikhonov_metrics:
            axes[1, 1].set_xlabel(f'MSE: {tikhonov_metrics["mse"]:.6f}')
        plt.colorbar(im3, ax=axes[1, 1])

        im4 = axes[1, 2].imshow(diff_cnn, cmap='Reds', vmin=0, vmax=0.5)
        axes[1, 2].set_title('Ошибка: CNN')
        if cnn_metrics:
            axes[1, 2].set_xlabel(f'MSE: {cnn_metrics["mse"]:.6f}')
        plt.colorbar(im4, ax=axes[1, 2])

        plt.suptitle('Сравнение методов реконструкции MPI', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_training_curves(train_losses, val_losses, save_path=None):
        """Построение кривых обучения"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('CNN Training Progress')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_summary_results(results, save_path=None):
        """Построение сводных графиков по результатам экспериментов"""
        if not results:
            print("Нет данных для построения графиков")
            return

        # Подготовка данных
        distances = [item['distance'] for item in results]
        tik_ssim = [item['results']['metrics']['tikhonov']['ssim'] for item in results]
        cnn_ssim = [item['results']['metrics']['cnn']['ssim'] for item in results]
        tik_psnr = [item['results']['metrics']['tikhonov']['psnr'] for item in results]
        cnn_psnr = [item['results']['metrics']['cnn']['psnr'] for item in results]
        tik_time = [item['results']['metrics']['tikhonov']['time'] for item in results]
        cnn_time = [item['results']['metrics']['cnn']['time'] for item in results]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # SSIM
        axes[0, 0].plot(distances, tik_ssim, 'ro-', label='Тихонов', linewidth=2, markersize=8)
        axes[0, 0].plot(distances, cnn_ssim, 'bs-', label='CNN', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Расстояние между каплями')
        axes[0, 0].set_ylabel('SSIM')
        axes[0, 0].set_title('Сравнение SSIM')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].invert_xaxis()

        # PSNR
        axes[0, 1].plot(distances, tik_psnr, 'ro-', label='Тихонов', linewidth=2, markersize=8)
        axes[0, 1].plot(distances, cnn_psnr, 'bs-', label='CNN', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Расстояние между каплями')
        axes[0, 1].set_ylabel('PSNR (дБ)')
        axes[0, 1].set_title('Сравнение PSNR')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].invert_xaxis()

        # Время
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

        # Улучшение
        ssim_improvement = [cnn - tik for tik, cnn in zip(tik_ssim, cnn_ssim)]
        psnr_improvement = [cnn - tik for tik, cnn in zip(tik_psnr, cnn_psnr)]

        axes[1, 1].bar(x_indices - bar_width / 2, ssim_improvement, bar_width,
                       label='ΔSSIM', color='green', alpha=0.7)
        axes[1, 1].bar(x_indices + bar_width / 2, psnr_improvement, bar_width,
                       label='ΔPSNR (дБ)', color='orange', alpha=0.7)
        axes[1, 1].set_xlabel('Расстояние между каплями')
        axes[1, 1].set_ylabel('Улучшение')
        axes[1, 1].set_title('Улучшение CNN vs Тихонов')
        axes[1, 1].set_xticks(x_indices)
        axes[1, 1].set_xticklabels([f'd={d}' for d in distances])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        plt.suptitle('Сводные результаты сравнения методов реконструкции MPI',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()