# Теоретические Основы MPI

## Основное уравнение MPI

Согласно работе [@sun2024image], измеряемый сигнал в MPI описывается
уравнением:

$$\begin{equation}
u(t) = \int_{\Omega} a(\mathbf{r}, t) c(\mathbf{r}) \, d\mathbf{r}
\label{eq:main_mpi}
\end{equation}$$

где:

- $u(t)$ -- измеряемый сигнал в приемной катушке

- $a(\mathbf{r}, t)$ -- системная функция MPI

- $c(\mathbf{r})$ -- пространственное распределение концентрации частиц

- $\Omega$ -- область визуализации

## Системная функция MPI

Системная функция определяется как [@sun2024image]:

$$\begin{equation}
a(\mathbf{r}, t) = -\mu_0 s(\mathbf{r}) \frac{\partial m(\mathbf{r}, t)}{\partial t}
\label{eq:system_function}
\end{equation}$$

где:

- $\mu_0$ -- магнитная постоянная

- $s(\mathbf{r})$ -- чувствительность приемной катушки

- $m(\mathbf{r}, t)$ -- намагниченность частиц

## Функция Ланжевена

Намагниченность частиц описывается функцией Ланжевена:

$$\begin{equation}
m(H) = m_{\text{sat}} \mathcal{L}\left(\frac{\mu_0 m_{\text{sat}} H}{k_B T}\right)
\label{eq:langevin}
\end{equation}$$

где функция Ланжевена определяется как:

$$\begin{equation}
\mathcal{L}(\xi) = \coth(\xi) - \frac{1}{\xi}
\label{eq:langevin_func}
\end{equation}$$

Для численной стабильности используются приближения: $$\begin{equation}
\mathcal{L}(\xi) \approx 
\begin{cases}
\frac{\xi}{3}  & \text{для } |\xi| \ll 1 \\
1 - \frac{1}{\xi} & \text{для } |\xi| \gg 1
\end{cases}
\label{eq:langevin_approx}
\end{equation}$$

## Магнитные поля в MPI

Полное магнитное поле состоит из двух компонентов:

$$\begin{equation}
\mathbf{H}_{\text{total}}(\mathbf{r}, t) = \mathbf{H}_{\text{sel}}(\mathbf{r} - \mathbf{r}_{\text{FFP}}(t)) + \mathbf{H}_{\text{exc}}(t)
\label{eq:total_field}
\end{equation}$$

### Селективное поле

Селективное поле создает Field-Free Point (FFP):

$$\begin{equation}
\mathbf{H}_{\text{sel}}(\mathbf{r}) = \mathbf{G} \cdot \mathbf{r} = \begin{bmatrix}
G_x & 0 \\
0 & G_y
\end{bmatrix} \begin{bmatrix}
x \\ y
\end{bmatrix}
\label{eq:selection_field}
\end{equation}$$

### Возбуждающее поле

Возбуждающее поле вызывает отклик частиц:

$$\begin{equation}
\mathbf{H}_{\text{exc}}(t) = A_E \cos(2\pi f_E t) \mathbf{\hat{x}} + A_E \cos(2\pi f_E t) \mathbf{\hat{y}}
\label{eq:excitation_field}
\end{equation}$$

### Траектория FFP

Для 2D сканирования используется фигура Лиссажу:

$$\begin{equation}
\mathbf{r}_{\text{FFP}}(t) = \begin{bmatrix}
\frac{L_x}{2} \sin\left(2\pi f_x t\right) \\
\frac{L_y}{2} \sin\left(2\pi f_y t\right)
\end{bmatrix}
\label{eq:ffp_trajectory}
\end{equation}$$

# Численная Реализация

## Дискретизация области

Пространственная область дискретизируется равномерной сеткой:

$$\begin{equation}
x_i = -L_x/2 + i \cdot \Delta x, \quad i = 0, 1, \ldots, N_x-1
\label{eq:spatial_discretization}
\end{equation}$$

$$\begin{equation}
y_j = -L_y/2 + j \cdot \Delta y, \quad j = 0, 1, \ldots, N_y-1
\label{eq:spatial_discretization_y}
\end{equation}$$

где $\Delta x = L_x/(N_x-1)$, $\Delta y = L_y/(N_y-1)$.

## Численное интегрирование

Сигнал вычисляется методом трапеций:

$$\begin{equation}
u(t_k) = \sum_{i=0}^{N_x-2} \sum_{j=0}^{N_y-2} \frac{a(x_i,y_j,t_k)c(x_i,y_j) + a(x_{i+1},y_j,t_k)c(x_{i+1},y_j) + \cdots}{4} \Delta x \Delta y
\label{eq:numerical_integration}
\end{equation}$$

## Производная намагниченности

Производная вычисляется методом конечных разностей:

$$\begin{equation}
\frac{\partial m}{\partial t} \approx \frac{m(\mathbf{r}, t + \Delta t) - m(\mathbf{r}, t)}{\Delta t}
\label{eq:derivative}
\end{equation}$$

# Параметры Модели

::: {#tab:parameters}
  Параметр           Значение                   Описание
  ------------------ -------------------------- -----------------------
  $L_x, L_y$         0.02 м                     Размер области
  $G_x, G_y$         $50 \times 10^3$ А/м/м     Градиенты поля
  $A_E$              0.01 А/м                   Амплитуда возбуждения
  $f_E$              10 кГц                     Частота возбуждения
  $m_{\text{sat}}$   $4 \times 10^{-18}$ А·м²   Магнитный момент
  $T$                300 K                      Температура
  $N_x, N_y$         50                         Разрешение сетки
  $N_t$              $N_x*N_y*3 = 7500$         Число временных точек
  $Time$             1 c                        Время сканирования

  : Параметры 2D MPI симуляции
:::

[]{#tab:parameters label="tab:parameters"}

# Алгоритм Вычислений

## Основной цикл симуляции

1.  **Инициализация сетки**: Создание пространственно-временной сетки

2.  **Определение концентрации**: Задание тестового распределения частиц

3.  **Вычисление системной функции**: Для каждого момента времени

4.  **Интегрирование сигнала**: Вычисление $u(t)$ по всей области

5.  **Визуализация**: Построение графиков результатов

## Ключевые функции

### Функция Ланжевена

Реализация включает три режима для численной стабильности:
$$\begin{equation}
\text{langevin}(H) = 
\begin{cases}
m_{\text{sat}} \cdot \frac{\xi}{3} & \xi < 10^{-6} \\
m_{\text{sat}} \cdot \left(1 - \frac{1}{\xi}\right) & \xi > 20 \\
m_{\text{sat}} \cdot \left(\coth(\xi) - \frac{1}{\xi}\right) & \text{иначе}
\end{cases}
\label{eq:numerical_langevin}
\end{equation}$$

### Системная функция

$$\begin{equation}
a(x,y,t) = -\mu_0 \cdot s(x,y) \cdot \frac{m(x,y,t+\Delta t) - m(x,y,t)}{\Delta t}
\label{eq:numerical_system_func}
\end{equation}$$

где $s(x,y)$ -- чувствительность катушки: $$\begin{equation}
s(x,y) = \frac{1}{1 + \left(\frac{\sqrt{x^2 + y^2}}{R_{\text{coil}}}\right)^2}
\label{eq:coil_sensitivity}
\end{equation}$$

## Результаты

<figure id="fig:FFP_lissaju" data-latex-placement="H">

<figcaption>Траектория FFP Лиссажу</figcaption>
</figure>

<figure id="fig:FFP_raster" data-latex-placement="H">

<figcaption>Траектория FFP растровый тип</figcaption>
</figure>

## Генерация синтетических данных с использованием готовой системной матрицы

<figure id="fig:circle" data-latex-placement="H">

<figcaption>Круг</figcaption>
</figure>

<figure id="fig:three_dots" data-latex-placement="H">

<figcaption>Три точки</figcaption>
</figure>

<figure id="fig:gaussian_spots" data-latex-placement="H">

<figcaption>Гауссовы пятна</figcaption>
</figure>

<figure id="fig:Y" data-latex-placement="H">

<figcaption>Буква "У"</figcaption>
</figure>

## Две капли

Размеры изображений 51х51 пикселей. Радиусы и расстояния указаны в
относительных величинах (x, y от -1 до 1, то есть размеры изображений в
относительных единицах 2х2).

<figure data-latex-placement="H">

</figure>

<figure data-latex-placement="H">

</figure>

<figure data-latex-placement="H">

</figure>

<figure data-latex-placement="H">

</figure>

<figure data-latex-placement="H">

</figure>

<figure data-latex-placement="H">

</figure>

<figure data-latex-placement="H">

</figure>

<figure data-latex-placement="H">

</figure>

<figure data-latex-placement="H">

</figure>

<figure data-latex-placement="H">

</figure>

<figure data-latex-placement="H">

</figure>

<figure data-latex-placement="H">

</figure>

::: {#tab:experiment_results}
   **Радиус**   **Расстояние**   **SSIM**   **PSNR, дБ**   **FWHM, пикс**   **Отношение FWHM**
  ------------ ---------------- ---------- -------------- ---------------- --------------------
      0.40          0.400         0.7203        19.2           18.00              0.973
      0.40          0.200         0.7223        19.3           23.50              1.000
      0.40          0.100         0.7209        18.9           23.00              0.979
      0.30          0.300         0.7927        20.2           19.00              0.950
      0.30          0.150         0.8061        19.9           19.00              0.950
      0.30          0.075         0.8069        19.7           19.00              0.950
      0.20          0.200         0.8973        23.4           15.00              0.952
      0.20          0.100         0.9042        23.2           14.00              0.933
      0.20          0.050         0.9084        23.2           12.50              0.893
      0.10          0.100         0.8970        22.5            8.50              1.062
      0.10          0.050         0.9041        22.6            7.00              1.000
      0.10          0.025         0.9041        22.6            7.00              1.000

  : Результаты экспериментов по реконструкции двух кругов
:::

::: thebibliography
9 Sun S., Chen Y., Zhao M., Xu L., Zhong J. *Image Reconstruction for
Magnetic Particle Imaging Based on Sparse Representation and Deep
Learning*. IEEE Transactions on Instrumentation and Measurement, 2024.
:::

# Математические Выкладки

## Стабильность функции Ланжевена

Для малых $\xi$ используется разложение в ряд Тейлора:

$$\begin{equation}
\mathcal{L}(\xi) = \frac{\xi}{3} - \frac{\xi^3}{45} + \frac{2\xi^5}{945} - \cdots
\end{equation}$$

Для больших $\xi$ используется асимптотическое разложение:

$$\begin{equation}
\mathcal{L}(\xi) = 1 - \frac{1}{\xi} + \frac{1}{3\xi^3} - \cdots
\end{equation}$$
