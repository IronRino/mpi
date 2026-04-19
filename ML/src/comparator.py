"""Сравнение методов реконструкции MPI: Тихонов vs KatsMarc vs Chae vs DIP vs Shang vs PGNet vs DEQ-MPI vs CNN vs MoDL vs Diffusion vs Hybrid"""

import numpy as np
import time
import os
import h5py
from .models import (TikhonovReconstructor, KatsMarcAlgorithm, ChaeSingleLayerNN,
                     DeepImagePrior, ShangCNN, PGNet, DEQMPI)
from .metrics import MetricsCalculator
from .visualization import Visualization
import torch


class MPIReconstructionComparator:
    """Сравнение методов реконструкции"""

    def __init__(self, system_matrix_path=None):
        self.load_system_matrix(system_matrix_path)
        self.cnn_trainer = None
        self.modl_trainer = None
        self.diffusion_trainer = None
        self.hybrid_trainer = None

        # Новые модели
        self.chae_model = None
        self.dip_model = None
        self.shang_model = None
        self.pgnet_model = None
        self.deq_model = None
        self.katsmarc = None

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

        # Инициализация KatsMarc (классический алгоритм)
        # Источник: Kaczmarz, S. (1937)
        self.katsmarc = KatsMarcAlgorithm(self.SM)

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

    def set_chae_model(self, chae_model):
        """Установка модели Chae (2017)"""
        self.chae_model = chae_model

    def set_dip_model(self, dip_model):
        """Установка модели Deep Image Prior (Dittmer et al., 2020)"""
        self.dip_model = dip_model

    def set_shang_model(self, shang_model):
        """Установка модели Shang et al. (2020)"""
        self.shang_model = shang_model

    def set_pgnet_model(self, pgnet_model):
        """Установка модели PGNet (Wu et al., 2023)"""
        self.pgnet_model = pgnet_model

    def set_deq_model(self, deq_model):
        """Установка модели DEQ-MPI (Güngör et al., 2024)"""
        self.deq_model = deq_model

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

    # ====================================================================
    # МЕТОД 1: Chae (2017) - Однослойная полносвязная нейронная сеть
    # ====================================================================
    def chae_reconstruction(self, measurement):
        """Реконструкция методом Chae (2017)"""
        if self.chae_model is None:
            raise ValueError("Chae модель не установлена")

        real_part = measurement.real
        imag_part = measurement.imag
        meas_vector = np.concatenate([real_part.flatten(), imag_part.flatten()])
        meas_tensor = torch.tensor(meas_vector, dtype=torch.float32).unsqueeze(0)

        self.chae_model.eval()
        with torch.no_grad():
            reconstructed = self.chae_model(meas_tensor)

        reconstructed = reconstructed.numpy().reshape(self.image_shape)

        if reconstructed.max() > 0:
            reconstructed = reconstructed / reconstructed.max()

        return reconstructed

    # ====================================================================
    # МЕТОД 2: Dittmer et al. (2020) - Deep Image Prior (DIP)
    # ====================================================================
    def dip_reconstruction(self, measurement, n_iterations=500):
        """
        Реконструкция Deep Image Prior (Dittmer et al., 2020)
        Обучается на одном изображении для соответствия измерениям
        """
        if self.dip_model is None:
            raise ValueError("DIP модель не установлена")

        import torch.optim as optim

        real_part = measurement.real
        imag_part = measurement.imag
        meas_vector = np.concatenate([real_part.flatten(), imag_part.flatten()])
        meas_tensor = torch.tensor(meas_vector, dtype=torch.float32).unsqueeze(0)

        # Создаем копию модели для этого конкретного измерения
        dip_model = DeepImagePrior(self.image_shape, latent_dim=100)
        optimizer = optim.Adam(dip_model.parameters(), lr=0.01)

        # Генерируем латентный вектор
        latent_z = dip_model.generate_random_latent()

        # Оптимизация
        dip_model.train()
        for iteration in range(n_iterations):
            optimizer.zero_grad()

            # Генерация изображения
            generated_image = dip_model(latent_z)
            generated_flat = generated_image.view(1, -1)

            # Прямой оператор
            if hasattr(self, 'SM_extended'):
                A_tensor = torch.tensor(self.SM_extended, dtype=torch.float32)
                measurement_pred = generated_flat @ A_tensor.T
            else:
                # Упрощенная версия
                measurement_pred = generated_flat[:, :meas_tensor.shape[1]]

            # Loss: несоответствие измерениям
            loss = torch.mean((measurement_pred - meas_tensor) ** 2)
            loss.backward()
            optimizer.step()

            if iteration % 100 == 0:
                print(f"  DIP iteration {iteration}, loss: {loss.item():.6f}")

        # Финальная реконструкция
        dip_model.eval()
        with torch.no_grad():
            reconstructed = dip_model(latent_z)[0, 0].numpy()

        if reconstructed.max() > 0:
            reconstructed = reconstructed / reconstructed.max()

        return reconstructed

    # ====================================================================
    # МЕТОД 3: Shang et al. (2020) - CNN для улучшения разрешения
    # ====================================================================
    def shang_reconstruction(self, measurement):
        """Реконструкция методом Shang et al. (2020)"""
        if self.shang_model is None:
            raise ValueError("Shang модель не установлена")

        real_part = measurement.real
        imag_part = measurement.imag
        meas_vector = np.concatenate([real_part.flatten(), imag_part.flatten()])

        # Для Shang CNN нужен вход в виде изображения
        # Сначала делаем грубую реконструкцию через псевдообращение
        try:
            A_pinv = np.linalg.pinv(self.SM)
            initial_recon = (A_pinv @ meas_vector).reshape(self.image_shape)
        except:
            initial_recon = np.zeros(self.image_shape)

        input_tensor = torch.tensor(initial_recon, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.shang_model.eval()
        with torch.no_grad():
            reconstructed = self.shang_model(input_tensor)[0, 0].numpy()

        if reconstructed.max() > 0:
            reconstructed = reconstructed / reconstructed.max()

        return reconstructed

    # ====================================================================
    # МЕТОД 4: Wu et al. (2023) - PGNet (Projection Generation Network)
    # ====================================================================
    def pgnet_reconstruction(self, measurement):
        """Реконструкция методом PGNet (Wu et al., 2023)"""
        if self.pgnet_model is None:
            raise ValueError("PGNet модель не установлена")

        real_part = measurement.real
        imag_part = measurement.imag
        meas_vector = np.concatenate([real_part.flatten(), imag_part.flatten()])
        meas_tensor = torch.tensor(meas_vector, dtype=torch.float32).unsqueeze(0)

        self.pgnet_model.eval()
        with torch.no_grad():
            reconstructed = self.pgnet_model(meas_tensor)[0, 0].numpy()

        if reconstructed.shape != self.image_shape:
            from scipy import ndimage
            reconstructed = ndimage.zoom(reconstructed,
                                         (self.nx / reconstructed.shape[0],
                                          self.ny / reconstructed.shape[1]),
                                         order=1)

        if reconstructed.max() > 0:
            reconstructed = reconstructed / reconstructed.max()

        return reconstructed

    # ====================================================================
    # МЕТОД 5: Güngör et al. (2024) - DEQ-MPI
    # ====================================================================
    def deq_reconstruction(self, measurement):
        """Реконструкция методом DEQ-MPI (Güngör et al., 2024)"""
        if self.deq_model is None:
            raise ValueError("DEQ-MPI модель не установлена")

        real_part = measurement.real
        imag_part = measurement.imag
        meas_vector = np.concatenate([real_part.flatten(), imag_part.flatten()])
        meas_tensor = torch.tensor(meas_vector, dtype=torch.float32).unsqueeze(0)

        self.deq_model.eval()
        with torch.no_grad():
            reconstructed = self.deq_model(meas_tensor)[0, 0].numpy()

        if reconstructed.max() > 0:
            reconstructed = reconstructed / reconstructed.max()

        return reconstructed

    # ====================================================================
    # МЕТОД 6: KatsMarc (Алгоритм Кацмарца, 1937)
    # ====================================================================
    def katsmarc_reconstruction(self, measurement, n_iterations=10, relaxation=1.0):
        """Реконструкция алгоритмом Кацмарца (Kaczmarz, 1937)"""
        meas_vector = np.concatenate([measurement[0, :], measurement[1, :]])
        reconstructed = self.katsmarc.reconstruct(meas_vector, n_iterations, relaxation)
        reconstructed = reconstructed.reshape(self.image_shape)

        if reconstructed.max() > 0:
            reconstructed = reconstructed / reconstructed.max()

        return reconstructed

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
        meas_vector = np.concatenate([real_part.flatten(), imag_part.flatten()])

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
        print("-" * 90)

        original_image = self.generate_test_case(radius, distance)
        measurement = self.generate_measurement(original_image)

        results = {}

        # Список всех методов с их источниками
        methods = [
            ('Тихонов', self.tikhonov_reconstruction, "Tikhonov (1963)"),
            ('KatsMarc', lambda m: self.katsmarc_reconstruction(m, n_iterations=20), "Kaczmarz (1937)"),
            ('Chae(2017)', self.chae_reconstruction if self.chae_model else None, "Chae - Single Layer NN"),
            ('DIP(2020)', lambda m: self.dip_reconstruction(m, n_iterations=300) if self.dip_model else None, "Dittmer et al. - Deep Image Prior"),
            ('Shang(2020)', self.shang_reconstruction if self.shang_model else None, "Shang et al. - CNN SR"),
            ('PGNet(2023)', self.pgnet_reconstruction if self.pgnet_model else None, "Wu et al. - PGNet"),
            ('DEQ-MPI(2024)', self.deq_reconstruction if self.deq_model else None, "Güngör et al. - DEQ-MPI"),
            ('CNN', self.cnn_reconstruction if self.cnn_trainer else None, "CNN (UNet)"),
            ('MoDL', self.modl_reconstruction if self.modl_trainer else None, "MoDL Network"),
            ('Diffusion', self.diffusion_reconstruction if self.diffusion_trainer else None, "Diffusion Model"),
            ('Hybrid', self.hybrid_reconstruction if self.hybrid_trainer else None, "Hybrid (MoDL+Diffusion)"),
        ]

        print(f"\n{'Метод':<15} {'Источник':<35} {'SSIM':<8} {'PSNR (дБ)':<12} {'FWHM':<8} {'Время (с)':<10}")
        print("-" * 90)

        for name, method, source in methods:
            if method is None:
                continue

            print(f"Реконструкция методом {name}...")
            start_time = time.time()
            try:
                recon = method(measurement)
                elapsed_time = time.time() - start_time

                # Проверка на корректность результата
                if recon is None or np.isnan(recon).any() or np.isinf(recon).any():
                    print(f"  {name}: Ошибка - некорректный результат")
                    continue

                metrics = MetricsCalculator.calculate_all_metrics(original_image, recon)
                metrics['time'] = elapsed_time

                results[name] = {
                    'image': recon,
                    'metrics': metrics,
                    'source': source
                }

                print(f"{name:<15} {source:<35} {metrics['ssim']:<8.4f} {metrics['psnr']:<12.2f} "
                      f"{metrics['fwhm']:<8.2f} {metrics['time']:<10.4f}")
            except Exception as e:
                print(f"{name:<15} {source:<35} ОШИБКА: {str(e)[:50]}")
                continue

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
        save_path = f'./DATA/results/all_methods_r{radius}_d{distance}.png'

        Visualization.plot_all_methods_comparison(
            original, results, radius, distance,
            save_path=save_path, show_plot=False
        )

    def run_full_comparison(self, radius=0.2, distances=[0.2, 0.15, 0.1, 0.05]):
        """Запуск полного сравнения для разных расстояний"""
        print("=" * 90)
        print("ПОЛНОЕ СРАВНЕНИЕ МЕТОДОВ РЕКОНСТРУКЦИИ MPI")
        print("=" * 90)
        print("\nСравниваемые методы и их источники:")
        print("  1. Тихонов     - Tikhonov regularization (1963)")
        print("  2. KatsMarc    - Kaczmarz algorithm (1937) - ART")
        print("  3. Chae(2017)  - Single-layer fully connected network (ETRI Journal)")
        print("  4. DIP(2020)   - Deep Image Prior (Dittmer et al., arXiv)")
        print("  5. Shang(2020) - CNN for resolution improvement (Physics in Medicine & Biology)")
        print("  6. PGNet(2023) - Projection Generation Network (Wu et al., Medical Physics)")
        print("  7. DEQ-MPI(2024) - Deep Equilibrium Model (Güngör et al., IEEE TMI)")
        print("  8. CNN         - Standard UNet")
        print("  9. MoDL        - Model-based Deep Learning")
        print(" 10. Diffusion   - Diffusion model")
        print(" 11. Hybrid      - MoDL + Diffusion")
        print("=" * 90)

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

        print("\n" + "=" * 120)
        print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
        print("=" * 120)

        header = f"{'Расст.':<8} {'Метод':<15} {'Источник':<35} {'SSIM':<8} {'PSNR':<10} {'FWHM':<8} {'Время':<8}"
        print(header)
        print("-" * 120)

        for result in self.results:
            distance = result['distance']
            for name, data in result['results'].items():
                m = data['metrics']
                source = data.get('source', '')
                print(f"{distance:<8.3f} {name:<15} {source:<35} {m['ssim']:<8.4f} "
                      f"{m['psnr']:<10.2f} {m['fwhm']:<8.2f} {m['time']:<8.4f}")

        # Статистика
        print("\n" + "=" * 120)
        print("СТАТИСТИКА ПО ВСЕМ ЭКСПЕРИМЕНТАМ")
        print("=" * 120)

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
            f.write("=" * 120 + "\n")
            f.write("СРАВНЕНИЕ МЕТОДОВ РЕКОНСТРУКЦИИ MPI\n")
            f.write("=" * 120 + "\n\n")

            f.write("СПИСОК МЕТОДОВ И ИСТОЧНИКОВ:\n")
            f.write("-" * 60 + "\n")
            f.write("1. Тихонов     - Tikhonov regularization (1963)\n")
            f.write("2. KatsMarc    - Kaczmarz algorithm (1937) - ART\n")
            f.write("3. Chae(2017)  - Single-layer fully connected network (ETRI Journal)\n")
            f.write("4. DIP(2020)   - Deep Image Prior (Dittmer et al., arXiv)\n")
            f.write("5. Shang(2020) - CNN for resolution improvement (Physics in Medicine & Biology)\n")
            f.write("6. PGNet(2023) - Projection Generation Network (Wu et al., Medical Physics)\n")
            f.write("7. DEQ-MPI(2024) - Deep Equilibrium Model (Güngör et al., IEEE TMI)\n")
            f.write("8. CNN         - Standard UNet\n")
            f.write("9. MoDL        - Model-based Deep Learning\n")
            f.write("10. Diffusion  - Diffusion model\n")
            f.write("11. Hybrid     - MoDL + Diffusion\n")
            f.write("\n" + "=" * 120 + "\n\n")

            for result in self.results:
                distance = result['distance']
                f.write(f"ЭКСПЕРИМЕНТ: radius={result['radius']}, distance={distance}\n")
                f.write("-" * 90 + "\n")
                f.write(f"{'Метод':<15} {'Источник':<35} {'SSIM':<8} {'PSNR (дБ)':<12} {'FWHM':<8} {'Время (с)':<10}\n")
                f.write("-" * 90 + "\n")

                for name, data in result['results'].items():
                    m = data['metrics']
                    source = data.get('source', '')
                    f.write(f"{name:<15} {source:<35} {m['ssim']:<8.4f} {m['psnr']:<12.2f} "
                            f"{m['fwhm']:<8.2f} {m['time']:<10.4f}\n")
                f.write("\n")

        print(f"\nРезультаты сохранены в файл: {filename}")