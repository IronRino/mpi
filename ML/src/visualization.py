"""Визуализация результатов реконструкции"""

import matplotlib
matplotlib.use('Agg')  # Используем бэкенд без GUI для предотвращения ошибок HTTP
import matplotlib.pyplot as plt
import numpy as np


class Visualization:
    """Класс для визуализации результатов"""

    @staticmethod
    def plot_comparison(original, tikhonov_recon, cnn_recon,
                        radius=None, distance=None,
                        tikhonov_metrics=None, cnn_metrics=None,
                        save_path=None, show_plot=False):
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

        # Сохраняем в файл
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"График сохранен: {save_path}")

        # Показываем только если явно запрошено
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    @staticmethod
    def plot_training_curves(train_losses, val_losses, save_path=None, show_plot=False):
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
            print(f"График обучения сохранен: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_summary_results(results, save_path=None, show_plot=False):
        """Построение сводных графиков по результатам экспериментов"""
        if not results:
            print("Нет данных для построения графиков")
            return

        # Подготовка данных
        distances = [item['distance'] for item in results]

        # Собираем метрики для всех методов
        methods_data = {}
        for result in results:
            for name, data in result['results'].items():
                if name not in methods_data:
                    methods_data[name] = {'ssim': [], 'psnr': [], 'time': [], 'fwhm': []}
                methods_data[name]['ssim'].append(data['metrics']['ssim'])
                methods_data[name]['psnr'].append(data['metrics']['psnr'])
                methods_data[name]['time'].append(data['metrics']['time'])
                methods_data[name]['fwhm'].append(data['metrics']['fwhm'])

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        colors = {'Тихонов': 'red', 'CNN': 'blue', 'MoDL': 'green', 'Diffusion': 'orange', 'Hybrid': 'purple'}
        markers = {'Тихонов': 'o', 'CNN': 's', 'MoDL': '^', 'Diffusion': 'D', 'Hybrid': 'v'}

        # SSIM
        for name in methods_data:
            axes[0, 0].plot(distances, methods_data[name]['ssim'],
                           color=colors.get(name, 'gray'),
                           marker=markers.get(name, 'o'),
                           label=name, linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Расстояние между каплями')
        axes[0, 0].set_ylabel('SSIM')
        axes[0, 0].set_title('Сравнение SSIM')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].invert_xaxis()

        # PSNR
        for name in methods_data:
            axes[0, 1].plot(distances, methods_data[name]['psnr'],
                           color=colors.get(name, 'gray'),
                           marker=markers.get(name, 'o'),
                           label=name, linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Расстояние между каплями')
        axes[0, 1].set_ylabel('PSNR (дБ)')
        axes[0, 1].set_title('Сравнение PSNR')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].invert_xaxis()

        # Время
        bar_width = 0.25
        x_indices = np.arange(len(distances))
        for i, name in enumerate(methods_data):
            offset = (i - len(methods_data)/2) * bar_width
            axes[1, 0].bar(x_indices + offset, methods_data[name]['time'],
                          bar_width, label=name, alpha=0.8,
                          color=colors.get(name, 'gray'))
        axes[1, 0].set_xlabel('Расстояние между каплями')
        axes[1, 0].set_ylabel('Время реконструкции (с)')
        axes[1, 0].set_title('Сравнение времени выполнения')
        axes[1, 0].set_xticks(x_indices)
        axes[1, 0].set_xticklabels([f'd={d}' for d in distances])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # FWHM
        for name in methods_data:
            axes[1, 1].plot(distances, methods_data[name]['fwhm'],
                           color=colors.get(name, 'gray'),
                           marker=markers.get(name, 'o'),
                           label=name, linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('Расстояние между каплями')
        axes[1, 1].set_ylabel('FWHM (пиксели)')
        axes[1, 1].set_title('Сравнение пространственного разрешения')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].invert_xaxis()

        plt.suptitle('Сводные результаты сравнения методов реконструкции MPI',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Сводный график сохранен: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    @staticmethod
    def plot_all_methods_comparison(original, results, radius, distance, save_path=None, show_plot=False):
        """Визуализация сравнения всех методов"""
        n_methods = len(results)
        fig, axes = plt.subplots(2, n_methods + 1, figsize=(4 * (n_methods + 1), 10))

        vmin, vmax = 0, 1

        # Оригинал
        axes[0, 0].imshow(original, cmap='hot', vmin=vmin, vmax=vmax)
        axes[0, 0].set_title('Оригинал')
        axes[0, 0].set_xlabel(f'radius={radius}, distance={distance}')

        axes[1, 0].axis('off')

        # Результаты методов
        for idx, (name, data) in enumerate(results.items()):
            recon = data['image']
            metrics = data['metrics']

            axes[0, idx + 1].imshow(recon, cmap='hot', vmin=vmin, vmax=vmax)
            axes[0, idx + 1].set_title(name)
            axes[0, idx + 1].set_xlabel(f"SSIM: {metrics['ssim']:.3f}\nPSNR: {metrics['psnr']:.1f} дБ")

            # Разностное изображение
            diff = np.abs(original - recon)
            im = axes[1, idx + 1].imshow(diff, cmap='Reds', vmin=0, vmax=0.5)
            axes[1, idx + 1].set_title(f'Ошибка: {name}')
            axes[1, idx + 1].set_xlabel(f'MSE: {metrics["mse"]:.6f}')
            plt.colorbar(im, ax=axes[1, idx + 1])

        plt.suptitle(f'Сравнение методов реконструкции MPI (radius={radius}, distance={distance})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"График сравнения сохранен: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)