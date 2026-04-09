"""Сравнение методов реконструкции MPI"""

import numpy as np
import time
import os
import h5py
from .models import TikhonovReconstructor
from .metrics import MetricsCalculator
from .visualization import Visualization


class MPIReconstructionComparator:
    """Сравнение методов реконструкции: Тихонов vs CNN"""

    def __init__(self, system_matrix_path="""./../Chinese data/SystemMatrix.h5"""):
        self.load_system_matrix(system_matrix_path)
        self.cnn_trainer = None
        self.results = []

    def load_system_matrix(self, path):
        """Загрузка системной матрицы"""
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
        """Установка обученной CNN модели"""
        self.cnn_trainer = cnn_trainer

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
        """Реконструкция методом Тихонова"""
        return self.tikhonov_reconstructor.reconstruct(measurement, mu, kmax)

    def cnn_reconstruction(self, measurement):
        """Реконструкция с помощью CNN"""
        if self.cnn_trainer is None:
            raise ValueError("CNN модель не установлена")

        reconstructed = self.cnn_trainer.predict(measurement)
        reconstructed = reconstructed[0, 0, :, :]

        # Изменение размера до нужного
        if reconstructed.shape != self.image_shape:
            if reconstructed.shape[0] > self.nx:
                start_x = (reconstructed.shape[0] - self.nx) // 2
                reconstructed = reconstructed[start_x:start_x + self.nx, :]
            if reconstructed.shape[1] > self.ny:
                start_y = (reconstructed.shape[1] - self.ny) // 2
                reconstructed = reconstructed[:, start_y:start_y + self.ny]

            if reconstructed.shape[0] < self.nx or reconstructed.shape[1] < self.ny:
                from scipy import ndimage
                reconstructed = ndimage.zoom(reconstructed,
                                             (self.nx / reconstructed.shape[0],
                                              self.ny / reconstructed.shape[1]),
                                             order=1)

        if reconstructed.max() > 0:
            reconstructed = reconstructed / reconstructed.max()

        return reconstructed

    def compare_methods(self, radius=0.2, distance=0.2):
        """Сравнение методов для конкретного случая"""
        print(f"\nСравнение методов для radius={radius}, distance={distance}")
        print("-" * 50)

        original_image = self.generate_test_case(radius, distance)
        measurement = self.generate_measurement(original_image)

        # Тихонов
        print("Реконструкция методом Тихонова...")
        tikhonov_start = time.time()
        tikhonov_recon = self.tikhonov_reconstruction(measurement)
        tikhonov_time = time.time() - tikhonov_start

        # CNN
        print("Реконструкция методом CNN...")
        cnn_start = time.time()
        cnn_recon = self.cnn_reconstruction(measurement)
        cnn_time = time.time() - cnn_start

        # Метрики
        tikhonov_metrics = MetricsCalculator.calculate_all_metrics(original_image, tikhonov_recon)
        cnn_metrics = MetricsCalculator.calculate_all_metrics(original_image, cnn_recon)

        # Добавляем время
        tikhonov_metrics['time'] = tikhonov_time
        cnn_metrics['time'] = cnn_time

        # Вывод
        print(f"\nРезультаты:")
        print(f"{'Метод':<10} {'SSIM':<8} {'PSNR (дБ)':<12} {'FWHM':<8} {'Время (с)':<10}")
        print("-" * 50)
        print(f"{'Тихонов':<10} {tikhonov_metrics['ssim']:<8.4f} "
              f"{tikhonov_metrics['psnr']:<12.2f} {tikhonov_metrics['fwhm']:<8.2f} "
              f"{tikhonov_metrics['time']:<10.4f}")
        print(f"{'CNN':<10} {cnn_metrics['ssim']:<8.4f} "
              f"{cnn_metrics['psnr']:<12.2f} {cnn_metrics['fwhm']:<8.2f} "
              f"{cnn_metrics['time']:<10.4f}")

        # Визуализация
        Visualization.plot_comparison(
            original_image, tikhonov_recon, cnn_recon,
            radius, distance, tikhonov_metrics, cnn_metrics,
            save_path=f'./DATA/results/comparison_r{radius}_d{distance}.png'
        )

        return {
            'original': original_image,
            'tikhonov': tikhonov_recon,
            'cnn': cnn_recon,
            'metrics': {
                'tikhonov': tikhonov_metrics,
                'cnn': cnn_metrics
            }
        }

    def run_comparison_experiments(self, radius_list=[0.2], distance_list=[0.2, 0.1, 0.05]):
        """Запуск серии экспериментов"""
        print("=" * 60)
        print("СРАВНЕНИЕ МЕТОДОВ РЕКОНСТРУКЦИИ MPI")
        print("=" * 60)

        results = []

        for radius in radius_list:
            for distance in distance_list:
                if distance >= 0:
                    print(f"\n{'=' * 50}")
                    print(f"Эксперимент: радиус={radius}, расстояние={distance}")
                    print(f"{'=' * 50}")

                    result = self.compare_methods(radius, distance)
                    results.append({
                        'radius': radius,
                        'distance': distance,
                        'results': result
                    })

        self.results = results
        return results

    def print_summary(self):
        """Вывод сводной таблицы результатов"""
        if not self.results:
            print("Нет результатов для вывода")
            return

        print("\n" + "=" * 80)
        print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
        print("=" * 80)

        print(f"{'Radius':<8} {'Distance':<10} {'Метод':<10} {'SSIM':<8} "
              f"{'PSNR (дБ)':<12} {'FWHM':<8} {'Время (с)':<10}")
        print("-" * 80)

        for item in self.results:
            radius = item['radius']
            distance = item['distance']
            tik_metrics = item['results']['metrics']['tikhonov']
            cnn_metrics = item['results']['metrics']['cnn']

            print(f"{radius:<8.3f} {distance:<10.3f} {'Тихонов':<10} "
                  f"{tik_metrics['ssim']:<8.4f} {tik_metrics['psnr']:<12.2f} "
                  f"{tik_metrics['fwhm']:<8.2f} {tik_metrics['time']:<10.4f}")

            print(f"{'':<8} {'':<10} {'CNN':<10} "
                  f"{cnn_metrics['ssim']:<8.4f} {cnn_metrics['psnr']:<12.2f} "
                  f"{cnn_metrics['fwhm']:<8.2f} {cnn_metrics['time']:<10.4f}")
            print("-" * 80)

        # Статистика
        print("\n" + "=" * 80)
        print("СТАТИСТИКА ПО ВСЕМ ЭКСПЕРИМЕНТАМ")
        print("=" * 80)

        tik_ssim_avg = np.mean([r['results']['metrics']['tikhonov']['ssim'] for r in self.results])
        cnn_ssim_avg = np.mean([r['results']['metrics']['cnn']['ssim'] for r in self.results])

        tik_psnr_avg = np.mean([r['results']['metrics']['tikhonov']['psnr'] for r in self.results])
        cnn_psnr_avg = np.mean([r['results']['metrics']['cnn']['psnr'] for r in self.results])

        tik_time_avg = np.mean([r['results']['metrics']['tikhonov']['time'] for r in self.results])
        cnn_time_avg = np.mean([r['results']['metrics']['cnn']['time'] for r in self.results])

        print(f"Средний SSIM:")
        print(f"  Тихонов: {tik_ssim_avg:.4f}")
        print(f"  CNN:     {cnn_ssim_avg:.4f}")
        if tik_ssim_avg > 0:
            print(f"  Улучшение: +{(cnn_ssim_avg - tik_ssim_avg):.4f} "
                  f"({(cnn_ssim_avg / tik_ssim_avg - 1) * 100:.1f}%)")

        print(f"\nСредний PSNR:")
        print(f"  Тихонов: {tik_psnr_avg:.2f} дБ")
        print(f"  CNN:     {cnn_psnr_avg:.2f} дБ")
        print(f"  Улучшение: +{(cnn_psnr_avg - tik_psnr_avg):.2f} дБ")

        print(f"\nСреднее время реконструкции:")
        print(f"  Тихонов: {tik_time_avg:.4f} с")
        print(f"  CNN:     {cnn_time_avg:.4f} с")
        if cnn_time_avg > 0:
            print(f"  Ускорение: {tik_time_avg / cnn_time_avg:.1f}x")

    def save_results_to_file(self, filename='./DATA/results/summary_results.txt'):
        """Сохранение результатов в файл"""
        if not self.results:
            print("Нет результатов для сохранения")
            return

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("РЕЗУЛЬТАТЫ СРАВНЕНИЯ МЕТОДОВ РЕКОНСТРУКЦИИ MPI\n")
            f.write("=" * 80 + "\n\n")

            for item in self.results:
                radius = item['radius']
                distance = item['distance']
                tik_metrics = item['results']['metrics']['tikhonov']
                cnn_metrics = item['results']['metrics']['cnn']

                f.write(f"ЭКСПЕРИМЕНТ: radius={radius}, distance={distance}\n")
                f.write("-" * 60 + "\n")
                f.write(f"{'Метод':<10} {'SSIM':<8} {'PSNR (дБ)':<12} "
                        f"{'FWHM':<8} {'Время (с)':<10}\n")
                f.write(f"{'Тихонов':<10} {tik_metrics['ssim']:<8.4f} "
                        f"{tik_metrics['psnr']:<12.2f} {tik_metrics['fwhm']:<8.2f} "
                        f"{tik_metrics['time']:<10.4f}\n")
                f.write(f"{'CNN':<10} {cnn_metrics['ssim']:<8.4f} "
                        f"{cnn_metrics['psnr']:<12.2f} {cnn_metrics['fwhm']:<8.2f} "
                        f"{cnn_metrics['time']:<10.4f}\n\n")

            # Итоговая статистика
            f.write("\n" + "=" * 80 + "\n")
            f.write("ОБЩАЯ СТАТИСТИКА\n")
            f.write("=" * 80 + "\n\n")

            tik_ssim_avg = np.mean([r['results']['metrics']['tikhonov']['ssim'] for r in self.results])
            cnn_ssim_avg = np.mean([r['results']['metrics']['cnn']['ssim'] for r in self.results])

            tik_psnr_avg = np.mean([r['results']['metrics']['tikhonov']['psnr'] for r in self.results])
            cnn_psnr_avg = np.mean([r['results']['metrics']['cnn']['psnr'] for r in self.results])

            tik_time_avg = np.mean([r['results']['metrics']['tikhonov']['time'] for r in self.results])
            cnn_time_avg = np.mean([r['results']['metrics']['cnn']['time'] for r in self.results])

            f.write(f"Средний SSIM:\n")
            f.write(f"  Тихонов: {tik_ssim_avg:.4f}\n")
            f.write(f"  CNN:     {cnn_ssim_avg:.4f}\n")
            if tik_ssim_avg > 0:
                f.write(f"  Улучшение: +{(cnn_ssim_avg - tik_ssim_avg):.4f} "
                        f"({(cnn_ssim_avg / tik_ssim_avg - 1) * 100:.1f}%)\n\n")

            f.write(f"Средний PSNR:\n")
            f.write(f"  Тихонов: {tik_psnr_avg:.2f} дБ\n")
            f.write(f"  CNN:     {cnn_psnr_avg:.2f} дБ\n")
            f.write(f"  Улучшение: +{(cnn_psnr_avg - tik_psnr_avg):.2f} дБ\n\n")

            f.write(f"Среднее время реконструкции:\n")
            f.write(f"  Тихонов: {tik_time_avg:.4f} с\n")
            f.write(f"  CNN:     {cnn_time_avg:.4f} с\n")
            if cnn_time_avg > 0:
                f.write(f"  Ускорение: {tik_time_avg / cnn_time_avg:.1f}x\n")

        print(f"\nРезультаты сохранены в файл: {filename}")

    def plot_summary_results(self, save_path='./DATA/results/summary_plot.png'):
        """Построение сводных графиков"""
        Visualization.plot_summary_results(self.results, save_path)