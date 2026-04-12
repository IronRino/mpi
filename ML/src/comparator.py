"""Сравнение методов реконструкции MPI: Тихонов vs CNN vs MoDL vs Diffusion vs Hybrid"""

import numpy as np
import time
import os
import h5py
from .models import TikhonovReconstructor
from .metrics import MetricsCalculator
from .visualization import Visualization


class MPIReconstructionComparator:
    """Сравнение методов реконструкции"""

    def __init__(self, system_matrix_path=None):
        self.load_system_matrix(system_matrix_path)
        self.cnn_trainer = None
        self.modl_trainer = None
        self.diffusion_trainer = None
        self.hybrid_trainer = None
        self.results = []

    def load_system_matrix(self, path):
        """Загрузка системной матрицы"""
        if path is None:
            possible_paths = [
                './DATA/SystemMatrix.h5',
                '../DATA/SystemMatrix.h5',
                './SystemMatrix.h5',
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    path = p
                    break
            else:
                raise FileNotFoundError("Не найден файл SystemMatrix.h5")

        fSM = h5py.File(path, 'r')
        S_data_r = fSM['/measurement/data/r'][:]
        S_data_i = fSM['/measurement/data/i'][:]
        S = S_data_r + 1j * S_data_i
        isBG = fSM['/measurement/isBackgroundFrame'][:].squeeze()
        S = S[:, :, isBG == 0]
        self.SM = S.reshape(S.shape[0] * S.shape[1], S.shape[2])

        number_Position = fSM['/calibration/size'][:].squeeze()
        self.nx, self.ny = int(number_Position[0]), int(number_Position[1])
        self.image_shape = (self.nx, self.ny)

        fSM.close()

        self.tikhonov_reconstructor = TikhonovReconstructor(self.SM)

        print(f"Размер изображения: {self.image_shape}")
        print(f"Размер системной матрицы: {self.SM.shape}")

    def set_cnn_model(self, cnn_trainer):
        self.cnn_trainer = cnn_trainer

    def set_modl_model(self, modl_trainer):
        self.modl_trainer = modl_trainer

    def set_diffusion_model(self, diffusion_trainer):
        self.diffusion_trainer = diffusion_trainer

    def set_hybrid_model(self, hybrid_trainer):
        self.hybrid_trainer = hybrid_trainer

    def generate_test_case(self, radius=0.2, distance=0.2, intensity1=0.7, intensity2=0.7):
        """Генерация тестового случая с двумя каплями"""
        image = np.zeros((self.nx, self.ny))

        x = np.linspace(-1, 1, self.nx)
        y = np.linspace(-1, 1, self.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        center1_x = -distance / 2
        center2_x = distance / 2

        sigma = radius / 2.5

        r1 = np.sqrt((X - center1_x) ** 2 + Y ** 2)
        image += intensity1 * np.exp(-(r1 ** 2) / (2 * sigma ** 2))

        r2 = np.sqrt((X - center2_x) ** 2 + Y ** 2)
        image += intensity2 * np.exp(-(r2 ** 2) / (2 * sigma ** 2))

        if image.max() > 0:
            image = image / image.max()

        return image

    def generate_measurement(self, image):
        """Генерация измерений из изображения"""
        image_vector = image.reshape(-1, 1)
        measurement = self.SM @ image_vector
        measurement_reshaped = measurement.reshape(2, -1)
        return measurement_reshaped

    def tikhonov_reconstruction(self, measurement, mu=0.001, kmax=100):
        return self.tikhonov_reconstructor.reconstruct(measurement, mu, kmax)

    def cnn_reconstruction(self, measurement):
        if self.cnn_trainer is None:
            raise ValueError("CNN модель не установлена")

        reconstructed = self.cnn_trainer.predict(measurement)

        # Обработка разных форматов вывода
        if isinstance(reconstructed, tuple):
            reconstructed = reconstructed[0]

        if reconstructed.ndim == 4:
            reconstructed = reconstructed[0, 0, :, :]
        elif reconstructed.ndim == 3:
            reconstructed = reconstructed[0, :, :]
        elif reconstructed.ndim == 2:
            pass  # уже правильная форма
        else:
            raise ValueError(f"Неожиданная форма вывода: {reconstructed.shape}")

        # Изменение размера до нужного
        if reconstructed.shape != self.image_shape:
            from scipy import ndimage
            reconstructed = ndimage.zoom(reconstructed,
                                         (self.nx / reconstructed.shape[0],
                                          self.ny / reconstructed.shape[1]),
                                         order=1)

        if reconstructed.max() > 0:
            reconstructed = reconstructed / reconstructed.max()

        return reconstructed

    def modl_reconstruction(self, measurement):
        """Реконструкция методом MoDL"""
        if self.modl_trainer is None:
            raise ValueError("MoDL модель не установлена")

        import torch

        # Измерения имеют форму (2, n_measurements)
        # Для MoDL нужно объединить реальную и мнимую части в вектор
        real_part = measurement.real  # (2, n_measurements)
        imag_part = measurement.imag  # (2, n_measurements)

        # Объединяем: сначала реальные части обеих катушек, затем мнимые
        # Размер: 2 * 2 * n_measurements = 4 * n_measurements
        # Но в системной матрице уже учтено, что M = 2550, а мы ожидаем 5100?
        # Проверим: real_part.flatten() имеет размер 2 * n_measurements
        # n_measurements = 1275 (так как SM имеет размер 2550, а reshape(2, -1) дает 2x1275)
        # Тогда meas_vector будет 4 * 1275 = 5100, что соответствует M=5100 в MoDL

        meas_vector = np.concatenate([real_part.flatten(), imag_part.flatten()])

        print(f"modl_reconstruction: measurement shape={measurement.shape}, "
              f"real_part shape={real_part.shape}, "
              f"meas_vector shape={meas_vector.shape}")

        meas_tensor = torch.tensor(meas_vector, dtype=torch.float32).unsqueeze(0)
        meas_tensor = meas_tensor.to(self.modl_trainer.device)

        self.modl_trainer.model.eval()
        with torch.no_grad():
            reconstructed = self.modl_trainer.model(meas_tensor)

        # Обработка вывода
        if isinstance(reconstructed, tuple):
            reconstructed = reconstructed[0]

        if reconstructed.ndim == 4:
            reconstructed = reconstructed[0, 0].cpu().numpy()
        elif reconstructed.ndim == 3:
            reconstructed = reconstructed[0].cpu().numpy()
        else:
            reconstructed = reconstructed.cpu().numpy()

        # Нормализация
        if reconstructed.max() > 0:
            reconstructed = reconstructed / reconstructed.max()

        # Изменение размера при необходимости
        if reconstructed.shape != self.image_shape:
            from scipy import ndimage
            reconstructed = ndimage.zoom(reconstructed,
                                         (self.nx / reconstructed.shape[0],
                                          self.ny / reconstructed.shape[1]),
                                         order=1)

        return reconstructed

    def diffusion_reconstruction(self, measurement):
        """Реконструкция методом диффузионной модели"""
        if self.diffusion_trainer is None:
            raise ValueError("Diffusion модель не установлена")

        import torch

        real_part = measurement.real
        imag_part = measurement.imag
        meas_vector = np.concatenate([real_part.flatten(), imag_part.flatten()])
        meas_tensor = torch.tensor(meas_vector, dtype=torch.float32).unsqueeze(0).to(self.diffusion_trainer.device)

        self.diffusion_trainer.model.eval()
        with torch.no_grad():
            # Для диффузионной модели нужен обратный процесс
            batch_size = meas_tensor.shape[0]
            x_t = torch.randn(batch_size, 1, self.nx, self.ny).to(self.diffusion_trainer.device)

            # Упрощенная версия обратного процесса
            for t in reversed(range(min(50, self.diffusion_trainer.model.n_steps))):
                t_tensor = torch.full((batch_size,), t, device=self.diffusion_trainer.device, dtype=torch.long)
                x_t = self.diffusion_trainer.model.p_sample(x_t, t_tensor)

            reconstructed = x_t[0, 0].cpu().numpy()

        if reconstructed.max() > 0:
            reconstructed = reconstructed / reconstructed.max()

        return reconstructed

    def hybrid_reconstruction(self, measurement):
        """Реконструкция гибридной моделью (MoDL + Diffusion)"""
        if self.hybrid_trainer is None:
            raise ValueError("Hybrid модель не установлена")

        import torch

        real_part = measurement.real
        imag_part = measurement.imag
        meas_vector = np.concatenate([real_part.flatten(), imag_part.flatten()])
        meas_tensor = torch.tensor(meas_vector, dtype=torch.float32).unsqueeze(0).to(self.hybrid_trainer.device)

        self.hybrid_trainer.model.eval()
        with torch.no_grad():
            reconstructed = self.hybrid_trainer.model(meas_tensor)

        if isinstance(reconstructed, tuple):
            reconstructed = reconstructed[0]

        reconstructed = reconstructed[0, 0].cpu().numpy()

        if reconstructed.max() > 0:
            reconstructed = reconstructed / reconstructed.max()

        if reconstructed.shape != self.image_shape:
            from scipy import ndimage
            reconstructed = ndimage.zoom(reconstructed,
                                         (self.nx / reconstructed.shape[0],
                                          self.ny / reconstructed.shape[1]),
                                         order=1)

        return reconstructed

    def compare_all_methods(self, radius=0.2, distance=0.2):
        """Сравнение всех доступных методов"""
        print(f"\nСравнение методов для radius={radius}, distance={distance}")
        print("-" * 70)

        original_image = self.generate_test_case(radius, distance)
        measurement = self.generate_measurement(original_image)

        results = {}
        methods = [
            ('Тихонов', self.tikhonov_reconstruction),
            ('CNN', self.cnn_reconstruction if self.cnn_trainer else None),
            ('MoDL', self.modl_reconstruction if self.modl_trainer else None),
            ('Diffusion', self.diffusion_reconstruction if self.diffusion_trainer else None),
            ('Hybrid', self.hybrid_reconstruction if self.hybrid_trainer else None),
        ]

        print(f"\n{'Метод':<12} {'SSIM':<8} {'PSNR (дБ)':<12} {'FWHM':<8} {'Время (с)':<10}")
        print("-" * 70)

        for name, method in methods:
            if method is None:
                continue

            print(f"Реконструкция методом {name}...")
            start_time = time.time()
            recon = method(measurement)
            elapsed_time = time.time() - start_time

            metrics = MetricsCalculator.calculate_all_metrics(original_image, recon)
            metrics['time'] = elapsed_time

            results[name] = {
                'image': recon,
                'metrics': metrics
            }

            print(f"{name:<12} {metrics['ssim']:<8.4f} {metrics['psnr']:<12.2f} "
                  f"{metrics['fwhm']:<8.2f} {metrics['time']:<10.4f}")

        # Визуализация
        self._visualize_all_comparison(original_image, results, radius, distance)

        return {
            'original': original_image,
            'results': results,
            'radius': radius,
            'distance': distance
        }

    def _visualize_all_comparison(self, original, results, radius, distance):
        """Визуализация сравнения всех методов"""
        from .visualization import Visualization

        save_path = f'./DATA/results/all_methods_r{radius}_d{distance}.png'

        Visualization.plot_all_methods_comparison(
            original, results, radius, distance,
            save_path=save_path, show_plot=False
        )

    def run_full_comparison(self, radius=0.2, distances=[0.2, 0.15, 0.1, 0.05]):
        """Запуск полного сравнения для разных расстояний"""
        print("=" * 70)
        print("ПОЛНОЕ СРАВНЕНИЕ МЕТОДОВ РЕКОНСТРУКЦИИ MPI")
        print("=" * 70)

        all_results = []

        for distance in distances:
            print(f"\n{'=' * 50}")
            print(f"Эксперимент: радиус={radius}, расстояние={distance}")
            print(f"{'=' * 50}")

            result = self.compare_all_methods(radius, distance)
            all_results.append(result)

        self.results = all_results
        return all_results

    def print_summary_table(self):
        """Вывод сводной таблицы результатов"""
        if not self.results:
            print("Нет результатов для вывода")
            return

        print("\n" + "=" * 100)
        print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
        print("=" * 100)

        header = f"{'Расст.':<8} {'Метод':<12} {'SSIM':<8} {'PSNR':<10} {'FWHM':<8} {'Время':<8}"
        print(header)
        print("-" * 100)

        for result in self.results:
            distance = result['distance']
            for name, data in result['results'].items():
                m = data['metrics']
                print(f"{distance:<8.3f} {name:<12} {m['ssim']:<8.4f} "
                      f"{m['psnr']:<10.2f} {m['fwhm']:<8.2f} {m['time']:<8.4f}")

        # Статистика
        print("\n" + "=" * 100)
        print("СТАТИСТИКА ПО ВСЕМ ЭКСПЕРИМЕНТАМ")
        print("=" * 100)

        # Собираем метрики по методам
        methods_metrics = {}
        for result in self.results:
            for name, data in result['results'].items():
                if name not in methods_metrics:
                    methods_metrics[name] = {'ssim': [], 'psnr': [], 'fwhm': [], 'time': []}
                methods_metrics[name]['ssim'].append(data['metrics']['ssim'])
                methods_metrics[name]['psnr'].append(data['metrics']['psnr'])
                methods_metrics[name]['fwhm'].append(data['metrics']['fwhm'])
                methods_metrics[name]['time'].append(data['metrics']['time'])

        for name, metrics in methods_metrics.items():
            print(f"\n{name}:")
            print(f"  Средний SSIM: {np.mean(metrics['ssim']):.4f} ± {np.std(metrics['ssim']):.4f}")
            print(f"  Средний PSNR: {np.mean(metrics['psnr']):.2f} ± {np.std(metrics['psnr']):.2f} дБ")
            print(f"  Средний FWHM: {np.mean(metrics['fwhm']):.2f} ± {np.std(metrics['fwhm']):.2f} пикс.")
            print(f"  Среднее время: {np.mean(metrics['time']):.4f} ± {np.std(metrics['time']):.4f} с")

    def save_results_to_file(self, filename='./DATA/results/all_methods_summary.txt'):
        """Сохранение результатов в файл"""
        if not self.results:
            print("Нет результатов для сохранения")
            return

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("СРАВНЕНИЕ МЕТОДОВ РЕКОНСТРУКЦИИ MPI\n")
            f.write("=" * 100 + "\n\n")

            for result in self.results:
                distance = result['distance']
                f.write(f"ЭКСПЕРИМЕНТ: radius={result['radius']}, distance={distance}\n")
                f.write("-" * 60 + "\n")
                f.write(f"{'Метод':<12} {'SSIM':<8} {'PSNR (дБ)':<12} {'FWHM':<8} {'Время (с)':<10}\n")
                f.write("-" * 60 + "\n")

                for name, data in result['results'].items():
                    m = data['metrics']
                    f.write(f"{name:<12} {m['ssim']:<8.4f} {m['psnr']:<12.2f} "
                            f"{m['fwhm']:<8.2f} {m['time']:<10.4f}\n")
                f.write("\n")

        print(f"\nРезультаты сохранены в файл: {filename}")