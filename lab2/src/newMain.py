import math
import matplotlib.pyplot as plot
import statistics as stat
import scipy.stats as stats
import numpy as np
from lab2.src import help

# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html

# =================================== Подготовка данных ===============================
f = open("../input/Task_2Kos.txt", 'r')
line = f.readline().split(" ")
data = []
data2 = [[], [], [], [], [], [], [], [], [], []]
numOfPoints = int(line[2])
numOfPointsInOneUnderArray = numOfPoints / 10
for t in f.readline().split(' '):
    data.append(float(t))
# создание 10 подвыборок
res = 0
for i in range(numOfPoints):
    j = int(i // numOfPointsInOneUnderArray)
    res += data[i]
    data2[j].append(data[i])

# сортировка значений
list.sort(data)
for l in data2:
    list.sort(l)

# ===========================  функция распределения и гистограммы =====================
m = 40  # кол-во интервалов
min_value = min(data)  # минимальное значение в выборке
max_value = max(data)  # максимальное значение в выборке
distribution_fun = np.zeros(m)

h = (max_value - min_value * 0.0000000000000001) / m  # шаг, с которым идут интервалы
# steps - точки на графике от 0.612678 до 12.9046 включительно, 40 элементов
steps = []  # массив 40 точек с шагом h
for t in range(1, m + 1):
    steps.append(min_value + t * h)

index = 0
for value in data:
    if value > steps[index]:
        p = int(abs(steps[index] - value) // h) + 1
        for i in range(1, p):
            distribution_fun[index + i] = distribution_fun[index]
        index += p
        distribution_fun[index] = distribution_fun[index - 1]
    distribution_fun[index] += 1
plot.title("Функция распределения")
plot.xlim([0.6, 13])
plot.bar(steps, distribution_fun / numOfPoints)
# plot.savefig("../newOut/destibutionFunction.png", dpi=200)
# plot.show()
plot.close()
plot.title("Гистограмма")
plot.hist(data, steps)
# plot.savefig("../newOut/histogram.png", dpi=200)
# plot.show()
plot.close()

# !!!!!!!!!Для относительной гистограммы
index = 0
for_relative = np.zeros(m)
for value in data:
    if value > steps[index]:
        p = int(abs(steps[index] - value) // h) + 1
        for_relative[index] = for_relative[index] / (h * numOfPoints)
        index += p
    for_relative[index] += 1
for_relative[m - 1] = for_relative[m - 1] / (h * numOfPoints)

# Проверка площади под гистограммой
ssss_____ = 0
for v in for_relative:
    ssss_____ += v * h
print('Area under the histogram : ', str(ssss_____))
# Конец проверки площади

plot.bar(steps, for_relative, width=h)
plot.xlim([0.6, 13])
plot.ylim([0, 0.3])
plot.title("Относительная гистограмма")
# plot.savefig("../newOut/relativeHistogram.png", dpi=200)
# plot.show()
plot.close()
# !!!!!!!!!!!!!!Относительная гистограмма построена

# ================== ТОЧЕЧНЫЕ ОЦЕНКИ =========================
print("================== ТОЧЕЧНЫЕ ОЦЕНКИ =========================")
empty = np.zeros(11)
median = [stat.median(data)]  # медианы
mean = [stat.mean(data)]  # среднее арифметическое (мат. ожидание)
mid_range = [(min_value + max_value) / 2]  # средина размаха
dispersion = [help.dispersion(data, mean[0])]  # дисперсия s^2
root_of_dispersion = [math.sqrt(dispersion[0])]  # корень из дисперсии s
third_central_moment = [help.central_moment(data, 3, mean[0])]  # 3-ий центральный момент
fourth_central_moment = [help.central_moment(data, 4, mean[0])]  # 4-ый центральный момент
asymmetry = [help.asymmetry(third_central_moment[0], root_of_dispersion[0])]  # асимметрия
kurtosis = [help.kurtosis(fourth_central_moment[0], dispersion[0])]  # эксцесса

interquantile_interval = help.interquantile_interval(numOfPoints, 0.5)  # интерквантильный интервал

index = 1
for n in data2:
    median.append(stat.median(n))
    mean.append(stat.mean(n))
    mid_range.append((min(n) + max(n)) / 2)
    dispersion.append(help.dispersion(data, mean[index]))
    root_of_dispersion.append((math.sqrt(dispersion[index])))
    third_central_moment.append(help.central_moment(data, 3, mean[index]))
    fourth_central_moment.append(help.central_moment(data, 4, mean[index]))
    asymmetry.append(third_central_moment[index] / pow(root_of_dispersion[index], 3))
    kurtosis.append(help.kurtosis(fourth_central_moment[index], dispersion[index]))
    index += 1
print('\tMin: ', min_value, ' Max: ', max_value)
print('\tx_med :', median)
print('\tM[x] :', mean)
print('\tx_ср :', mid_range)
print('\ts^2 :', dispersion)
print('\ts :', root_of_dispersion)
print('\t∘µ_3 :', third_central_moment)
print('\t∘µ_4 :', fourth_central_moment)
print('\tAs :', asymmetry)
print('\tEx :', kurtosis)
print('\tJ (номера значений) :', interquantile_interval)
print('\tJ (значения) :',
      "(" + str(data[interquantile_interval[0]]) + ", " + str(data[interquantile_interval[1] - 1]) + ")")

# ==================== ГРАФИКИ ТОЧЕЧНЫХ ПОКАЗАТЕЛЕЙ =========================
plot.figure()

ax1 = plot.subplot(9, 1, 1)
ax1.set_ylim(-0.1, 0.1)
ax1.set_yticks([])
ax1.set_yticklabels([])
plot.title('Медианы')
plot.plot(median, empty, 'r+')
plot.plot(median[0], 0, 'rp')

ax2 = plot.subplot(9, 1, 3)
ax2.set_yticklabels([])
ax2.set_yticks([])
plot.title('Среднее арифметическое (мат ожидание)')
plot.plot(mean, empty, 'b+')
plot.plot(mean[0], 0, 'bp')

ax3 = plot.subplot(9, 1, 5)
ax3.set_yticks([])
ax3.set_yticklabels([])
plot.title('Средина размаха')
plot.plot(mid_range, empty, 'g+')
plot.plot(mid_range[0], 0, 'gp')

ax4 = plot.subplot(9, 1, 7)
ax4.set_yticks([])
ax4.set_yticklabels([])
plot.title('Дисперсия')
plot.plot(dispersion, empty, 'g+')
plot.plot(dispersion[0], 0, 'gp')

ax5 = plot.subplot(9, 1, 9)
ax5.set_yticks([])
ax5.set_yticklabels([])
plot.title('Среднеквадратичное отклонение')
plot.plot(root_of_dispersion, empty, 'g+')
plot.plot(root_of_dispersion[0], 0, 'gp')

# plot.savefig("../newOut/moments1.png", dpi=200)
# plot.show()
plot.close()

plot.figure()
ax1 = plot.subplot(7, 1, 1)
ax1.set_ylim(-0.1, 0.1)
ax1.set_yticks([])
ax1.set_yticklabels([])
plot.title('Третий центральный момент')
plot.plot(third_central_moment, empty, 'r+')
plot.plot(third_central_moment[0], 0, 'rp')

ax2 = plot.subplot(7, 1, 3)
ax2.set_yticklabels([])
ax2.set_yticks([])
plot.title('Четвертый центральный момент')
plot.plot(fourth_central_moment, empty, 'b+')
plot.plot(fourth_central_moment[0], 0, 'bp')

ax3 = plot.subplot(7, 1, 5)
ax3.set_yticks([])
ax3.set_yticklabels([])
plot.title('Асимметрия')
plot.plot(asymmetry, empty, 'g+')
plot.plot(asymmetry[0], 0, 'gp')

ax4 = plot.subplot(7, 1, 7)
ax4.set_yticks([])
ax4.set_yticklabels([])
plot.title('Эксцесса')
plot.plot(kurtosis, empty, 'g+')
plot.plot(kurtosis[0], 0, 'gp')
# plot.savefig("../newOut/moments2.png", dpi=200)
# plot.show()
plot.close()
# ==================== ГРАФИКИ ТОЧЕЧНЫХ ПОКАЗАТЕЛЕЙ НАЧЕРЧЕНЫ =================

# ====================== Часть 1.4 . Интервальные оценки ==================
print("======================!!! Часть 1.4 . Интервальные оценки !!!===============")
Q = 0.8  # доверительная вероятность
left_chi2inv = 1.2400e+04  # посчитаны в MATLAB функцией chi2inv((1 + Q) / 2, n-1)
right_chi2inv = 1.1999e+04  # посчитаны в MATLAB функцией chi2inv((1 - Q) / 2, n-1)
tinv = 1.2816  # посчитано в MATLAB функцией tinv(0.9, n-1), 0.9 = (1+q)/2, где q=0.8
mean_interval = [help.mean_interval(numOfPoints, mean[0], root_of_dispersion[0], tinv)]
dispersion_interval = [help.dispersion_interval(numOfPoints, dispersion[0], left_chi2inv, right_chi2inv)]

for i in range(1, 11):
    mean_interval.append(help.mean_interval(numOfPoints, mean[i], root_of_dispersion[i], tinv))
    dispersion_interval.append(help.dispersion_interval(numOfPoints, dispersion[i], left_chi2inv, right_chi2inv))
print("\t Интервальные оценки для мат. ожидания" + str(mean_interval))
print("\t Интервальные оценки для дисперсии" + str(dispersion_interval))
# =================== Чертим ИНТЕРВАЛЬНЫЕ ОЦЕНКИ МАТ ОЖИДАНИЯ М ДИСПЕРСИИ ====================================
# Для мат. ожидания
plot.figure()
axes = [plot.subplot(11, 1, 1)]
axes[0].set_yticks([])
axes[0].set_ylabel('Full')
plot.title('Интервальные оценки мат. ожидания')
plot.setp(axes[0].get_xticklabels(), visible=False)
plot.plot(mean[0], 0, 'rp')
plot.plot(mean_interval[0][0], 0, 'b<')
plot.plot(mean_interval[0][1], 0, 'b>')

for i in range(1, 11):
    axes.append(plot.subplot(11, 1, i + 1, sharex=axes[0]))
    axes[i].set_yticks([])
    axes[i].set_ylabel(str(i))
    if i < 10: plot.setp(axes[i].get_xticklabels(), visible=False)
    plot.plot(mean[i], 0, 'r+')
    plot.plot(mean_interval[i][0], 0, 'b<')
    plot.plot(mean_interval[i][1], 0, 'b>')
axes[0].set_xlim([4.4, 4.6])
plot.savefig("../newOut/intervalsMoments.png", dpi=200)
plot.show()
plot.close()

# Для дисперсии

plot.figure()
axes = [plot.subplot(11, 1, 1)]
axes[0].set_yticks([])
axes[0].set_ylabel('Full')
plot.title('Интервальные оценки дисперсии')
plot.setp(axes[0].get_xticklabels(), visible=False)
plot.plot(dispersion[0], 0, 'rp')
plot.plot(dispersion_interval[0][0], 0, 'b<')
plot.plot(dispersion_interval[0][1], 0, 'b>')

for i in range(1, 11):
    axes.append(plot.subplot(11, 1, i + 1, sharex=axes[0]))
    axes[i].set_yticks([])
    axes[i].set_ylabel(str(i))
    if i < 10: plot.setp(axes[i].get_xticklabels(), visible=False)
    plot.plot(dispersion[i], 0, 'r+')
    plot.plot(dispersion_interval[i][0], 0, 'b<')
    plot.plot(dispersion_interval[i][1], 0, 'b>')
axes[0].set_xlim([2.18, 2.3])
plot.savefig("../newOut/intervalDispersion.png", dpi=200)
# plot.show()
plot.close()
# =================== графики ИНТЕРВАЛЬНЫХ ОЦЕНКИ МАТ ОЖИДАНИЯ М ДИСПЕРСИИ напечатаны! ==========================

# ============================= ТОЛЕРАНТНЫЕ ПРЕДЕЛЫ ===================================
print("============================= ТОЛЕРАНТНЫЕ ПРЕДЕЛЫ ===================================")
p = 0.95  # вероятность для интерквантильного промежутка
q = 0.8  # доверительная вероятность
tolerant_interval_average = [0, 0]  # массив для толерантных пределов

k = help.find_k(numOfPoints, p, q)  # кол-во отрасываемых точек
print("\tПредел k : " + str(k) + " , Значение биномиального распределения : " + str(
    stats.binom.cdf(numOfPoints - k, numOfPoints, p)))
# Для всей выборки относительно среднего арифметического
if k % 2 == 0:
    left_lim = int(k / 2)
    right_lim = int(numOfPoints - k / 2)
    tolerant_interval_average[0], tolerant_interval_average[1] = data[left_lim], data[right_lim]
else:
    left_lim = int((k - 1) / 2)
    right_lim = int(numOfPoints - (k - 1) / 2)
    tolerant_interval_average[0], tolerant_interval_average[1] = data[left_lim], data[right_lim]

# Для всей выборки относительно нуля
# Для этого возьмем модули отрицательных значений и пересортируем выборку
data_abs = np.sort(abs(np.array(data)))
tolerant_interval_zero = [-data_abs[numOfPoints - k + 1], data_abs[numOfPoints - k + 1]]
print("\tТолерантные пределы для всей выборки относительно среднего: " + str(tolerant_interval_average))
print("\tТолерантные пределы для всей выборки относительно нуля" + str(tolerant_interval_zero))

# ЧЕРТИМ
plot.title("Толерантные пределы для интерквантильного \nпромежутка относительно среднего значения")
plot.yticks([])
plot.plot(tolerant_interval_average[0], 0, 'b<')
plot.plot(tolerant_interval_average[1], 0, 'b>')
plot.plot(data[interquantile_interval[0]], 0, 'ro')
plot.plot(data[interquantile_interval[1]], 0, 'ro')
plot.legend(("Левый толерантный предел", "Правый толерантный предел", "Интерквантильный промежуток"), loc='upper right')
plot.savefig("../newOut/tolerantLimsAverage.png", dpi=200)
plot.show()
plot.close()

plot.title("Толерантные пределы относительно нуля")
plot.yticks([])
plot.plot(tolerant_interval_zero[0], 0, 'b<')
plot.plot(tolerant_interval_zero[1], 0, 'b>')
plot.legend(("Левый толерантный предел", "Правый толерантный предел"), loc='upper right')
plot.savefig("../newOut/tolerantLimsZero.png", dpi=200)
plot.show()
plot.close()

# Считаем параметрические толерантные пределы подвыборок
k_tolerant_multiplier = 1.96
parametric_tolerant_interval = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
for i in range(10):
    parametric_tolerant_interval[i][0] = mean[i + 1] - k_tolerant_multiplier * root_of_dispersion[i + 1]
    parametric_tolerant_interval[i][1] = mean[i + 1] + k_tolerant_multiplier * root_of_dispersion[i + 1]
print("\tПараметрические толерантные интервалы для подвыборок:")
print("\t\t" + str(parametric_tolerant_interval))

axes = []
plot.title("Параметрические толерантные пределы для подвыборок")
for i in range(10):
    if i == 0:
        axes.append(plot.subplot(10, 1, i + 1))
    else:
        axes.append(plot.subplot(10, 1, i + 1, sharex=axes[0]))
    axes[i].set_yticks([])
    axes[i].set_ylabel(str(i + 1))
    if i < 9: plot.setp(axes[i].get_xticklabels(), visible=False)
    plot.plot(parametric_tolerant_interval[i][0], 0, 'b<')
    plot.plot(parametric_tolerant_interval[i][1], 0, 'b>')
    plot.plot(mean[i + 1], 0, 'ro')
plot.savefig("../newOut/parametricTolerantLims.png", dpi=200)
plot.show()
plot.close()

# ============================= ЧАСТЬ 2 ========================================
# ========================== МЕТОД МОМЕНТОВ ====================================
print("===========================МЕТОД МОМЕНТОВ==========================")
# Релея, Гамма и Лапласа

# Для распредления Релея
sigma_for_rayleigh_moment_method = mean[0] / math.sqrt(math.pi / 2)
print("\tДля распределения Релея")
print("\t\tsigma = " + str(sigma_for_rayleigh_moment_method))

a_for_laplace_moment_method = median[0]
laplace_lambda_moment_method = math.sqrt(2 / dispersion[0])
print("\tДля распределения Лапласа")
print("\t\ta = " + str(a_for_laplace_moment_method) + " lambda = " + str(laplace_lambda_moment_method))

k_for_gamma_moment_method = (mean[0] ** 2) / dispersion[0]
theta_for_gamma_moment_method = dispersion[0] / mean[0]
print("\tДля Гамма-распредления")
print("\t\tk = " + str(k_for_gamma_moment_method) + " lambda = " + str(theta_for_gamma_moment_method))

# ======================================= ММП ====================================================
print("===========================ММП==========================")

# Для распределения Релея
dispersion_for_rayleigh_mmp = 1 / (2 * numOfPoints) * sum(np.array(data) ** 2)
s_for_rayleigh_mmp = math.sqrt(dispersion_for_rayleigh_mmp)
print("\tДля распределения Релея")
print("\t\tsigma = " + str(s_for_rayleigh_mmp))

# Для распределения Лапласа
a_for_laplace_mmp = mean[0]
laplace_lambda_mmp = numOfPoints * (1 / sum(abs(np.array(data) - a_for_laplace_mmp)))
print("\tДля распределения Лапласа")
print("\t\ta = " + str(a_for_laplace_mmp) + " lambda = " + str(laplace_lambda_mmp))

# Для Гамма-распределения
# Числовые значения, которые нужно посчитать
for_optimize1 = 0
for_optimize2 = 0
for v in data:
    if v > 0:
        for_optimize1 += v
        for_optimize2 += np.log(v)
for_optimize3 = for_optimize1
for_optimize1 = np.log(for_optimize1 / numOfPoints)
for_optimize2 = for_optimize2 / numOfPoints
c_mmp = for_optimize1 - for_optimize2

# Достаем градиент Гамма-функции и ищем ее минимум, числа 10 и 8, найдены чисто эмпирически
# 10 дает отрицательное значение функции, 8 положительное
gamma_gradient = help.gammaGradient(c_mmp).gamma_gradient
k_for_gamma_mmp = help.fmin_bisection(gamma_gradient, 10, 8, 1e-14)
theta_for_gamma_mmp = for_optimize3 / (k_for_gamma_mmp * numOfPoints)
print("\tДля Гамма-распределения")
print("\t\tk = " + str(k_for_gamma_mmp) + " theta = " + str(theta_for_gamma_mmp))

# ======================= Построим финции распределения и плотности вместе с гистограммой

# Для нормального распределения
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html#scipy.stats.norm
plot.title("Сравнение с плотностью распределения Рэлея")
plot.bar(steps, for_relative, width=h)
plot.xlim([0.6, 13])
plot.ylim([0, 0.3])
plot.plot(data, stats.rayleigh.pdf(np.array(data), scale=sigma_for_rayleigh_moment_method), 'b')
plot.plot(data, stats.rayleigh.pdf(np.array(data), scale=s_for_rayleigh_mmp), 'r')
plot.legend(("Метод моментов", "ММП", "Гистограмма"), loc='upper right')
# plot.savefig("../newOut/withRayleigh.png", dpi=200)
# plot.show()
plot.close()

plot.title("Сравнение с распределением Рэлея")
plot.bar(steps, distribution_fun / numOfPoints, width=h)
plot.plot(data, stats.rayleigh.cdf(np.array(data), scale=sigma_for_rayleigh_moment_method), 'b')
plot.plot(data, stats.rayleigh.cdf(np.array(data), scale=s_for_rayleigh_mmp), 'r')
plot.legend(("Метод моментов", "ММП", "Эмпирическая"), loc='upper right')
# plot.savefig("../newOut/withRayleighCumulative.png", dpi=200)
# plot.show()
plot.close()

# Для распределения Лапласа

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.laplace.html
plot.title("Сравнение с плотностью распределения Лапласа")
plot.bar(steps, for_relative, width=h)
plot.xlim([0.6, 13])
plot.ylim([0, 0.6])
plot.plot(data,
          stats.laplace.pdf(np.array(data), loc=a_for_laplace_moment_method, scale=1 / laplace_lambda_moment_method),
          'b')
plot.plot(data, stats.laplace.pdf(np.array(data), loc=a_for_laplace_mmp, scale=1 / laplace_lambda_mmp), 'r')
plot.legend(("Метод моментов", "ММП", "Гистограмма"), loc='upper right')
# plot.savefig("../newOut/withLaplace.png", dpi=200)
# plot.show()
plot.close()

plot.title("Сравнение с распределением Лапласа")
plot.bar(steps, distribution_fun / numOfPoints, width=h)
plot.plot(data, stats.laplace.cdf(np.array(data), loc=mean[0], scale=1 / laplace_lambda_moment_method), 'b')
plot.plot(data, stats.laplace.cdf(np.array(data), loc=a_for_laplace_mmp, scale=1 / laplace_lambda_mmp), 'r')
plot.legend(("Метод моментов", "ММП", "Эмпирическая"), loc='upper right')
# plot.savefig("../newOut/withLaplaceCumulative.png", dpi=200)
# plot.show()
plot.close()

# Для Гамма-распределения
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
plot.title("Сравнение с плотностью Гамма-распределения")
plot.bar(steps, for_relative, width=h)
plot.xlim([0.6, 13])
plot.ylim([0, 0.3])
plot.plot(data, stats.gamma.pdf(np.array(data), k_for_gamma_moment_method, scale=theta_for_gamma_moment_method), 'b')
plot.plot(data, stats.gamma.pdf(np.array(data), k_for_gamma_mmp, scale=theta_for_gamma_mmp), 'r')
plot.legend(("Метод моментов", "ММП", "Гистограмма"), loc='upper right')
# plot.savefig("../newOut/withGamma.png", dpi=200)
# plot.show()
plot.close()

plot.title("Сравнение с Гамма-распределением")
plot.bar(steps, distribution_fun / numOfPoints, width=h)
plot.plot(data, stats.gamma.cdf(np.array(data), k_for_gamma_moment_method, scale=theta_for_gamma_moment_method), 'b')
plot.plot(data, stats.gamma.cdf(np.array(data), k_for_gamma_mmp, scale=theta_for_gamma_mmp), 'r')
plot.legend(("Метод моментов", "ММП", "Эмпирическая"), loc='upper right')
# plot.savefig("../newOut/withGammaCumulative.png", dpi=200)
# plot.show()
plot.close()

# ========================== ПРОВЕРКА ГИПОТЕЗ ============================================
print("========================== ПРОВЕРКА ГИПОТЕЗ ==============================")
# _nk  - кол-во точек, попавших в k-ый интервал
_nk = np.empty(m)
index = 0
for val in distribution_fun:
    if index == 0:
        _nk[index] = val
    else:
        _nk[index] = val - distribution_fun[index - 1]
    index += 1

# =============== Хи-квадрат==============================================================
print("=============== Хи-квадрат статистика=====================")
print("\tКритическое значение для Рэлея = 50.6598")
print("\tКритическое значение для Лапласа и Гаммы = 49.5126")  # Значение получено в MATLAB
print("\tДля распределения Рэлея")
index = 0
chi2_stat = 0
for i in range(40):
    if i == 0:
        ___Pk = stats.rayleigh.cdf(steps[index], scale=sigma_for_rayleigh_moment_method) - \
                stats.rayleigh.cdf(min_value, scale=sigma_for_rayleigh_moment_method)
    else:
        ___Pk = stats.rayleigh.cdf(steps[index], scale=sigma_for_rayleigh_moment_method) - \
                stats.rayleigh.cdf(steps[index - 1], scale=sigma_for_rayleigh_moment_method)
    chi2_stat += (numOfPoints * ___Pk - _nk[index]) ** 2 / (numOfPoints * ___Pk)
    index += 1
print("\t\tДля метода моментов = " + str(chi2_stat))

index = 0
chi2_stat = 0
for i in range(40):
    if i == 0:
        ___Pk = stats.rayleigh.cdf(steps[index], scale=s_for_rayleigh_mmp) - \
                stats.rayleigh.cdf(min_value, scale=s_for_rayleigh_mmp)
    else:
        ___Pk = stats.rayleigh.cdf(steps[index], scale=s_for_rayleigh_mmp) - \
                stats.rayleigh.cdf(steps[index - 1], scale=s_for_rayleigh_mmp)
    chi2_stat += (numOfPoints * ___Pk - _nk[index]) ** 2 / (numOfPoints * ___Pk)
    index += 1
print("\t\tДля ММП = " + str(chi2_stat))

print("\tДля распределения Лапласа")
index = 0
chi2_stat = 0
for i in range(40):
    if i == 0:
        ___Pk = stats.laplace.cdf(steps[index], loc=a_for_laplace_moment_method,
                                  scale=1 / laplace_lambda_moment_method) - \
                stats.laplace.cdf(min_value, loc=a_for_laplace_moment_method, scale=1 / laplace_lambda_moment_method)
    else:
        ___Pk = stats.laplace.cdf(steps[index], loc=a_for_laplace_moment_method,
                                  scale=1 / laplace_lambda_moment_method) - \
                stats.laplace.cdf(steps[index - 1], loc=a_for_laplace_moment_method,
                                  scale=1 / laplace_lambda_moment_method)
    chi2_stat += (numOfPoints * ___Pk - _nk[index]) ** 2 / (numOfPoints * ___Pk)
    index += 1
print("\t\tДля метода моментов = " + str(chi2_stat))

index = 0
chi2_stat = 0
for i in range(40):
    if i == 0:
        ___Pk = stats.laplace.cdf(steps[index], loc=a_for_laplace_mmp, scale=1 / laplace_lambda_mmp) - \
                stats.laplace.cdf(min_value, loc=a_for_laplace_mmp, scale=1 / laplace_lambda_mmp)
    else:
        ___Pk = stats.laplace.cdf(steps[index], loc=a_for_laplace_mmp, scale=1 / laplace_lambda_mmp) - \
                stats.laplace.cdf(steps[index - 1], loc=a_for_laplace_mmp, scale=1 / laplace_lambda_mmp)
    chi2_stat += (numOfPoints * ___Pk - _nk[index]) ** 2 / (numOfPoints * ___Pk)
    index += 1
print("\t\tДля ММП = " + str(chi2_stat))

print("\tДля Гамма-распределения")
index = 0
chi2_stat = 0
for i in range(40):
    if i == 0:
        ___Pk = stats.gamma.cdf(steps[index], k_for_gamma_moment_method, scale=theta_for_gamma_moment_method) - \
                stats.gamma.cdf(min_value, k_for_gamma_moment_method, scale=theta_for_gamma_moment_method)
    else:
        ___Pk = stats.gamma.cdf(steps[index], k_for_gamma_moment_method, scale=theta_for_gamma_moment_method) - \
                stats.gamma.cdf(steps[index - 1], k_for_gamma_moment_method, scale=theta_for_gamma_moment_method)
    chi2_stat += (numOfPoints * ___Pk - _nk[index]) ** 2 / (numOfPoints * ___Pk)
    index += 1
print("\t\tДля метода моментов = " + str(chi2_stat))

index = 0
chi2_stat = 0
for i in range(40):
    if i == 0:
        ___Pk = stats.gamma.cdf(steps[index], k_for_gamma_moment_method, scale=theta_for_gamma_moment_method) - \
                stats.gamma.cdf(min_value, k_for_gamma_moment_method, scale=theta_for_gamma_moment_method)
    else:
        ___Pk = stats.gamma.cdf(steps[index], k_for_gamma_mmp, scale=theta_for_gamma_mmp) - \
                stats.gamma.cdf(steps[index - 1], k_for_gamma_mmp, scale=theta_for_gamma_mmp)
    chi2_stat += (numOfPoints * ___Pk - _nk[index]) ** 2 / (numOfPoints * ___Pk)
    index += 1
print("\t\tДля ММП = " + str(chi2_stat))

# =============== КОЛМАГОРОВА - СМИРНОВА==============================================================
print("=============== статистика КОЛМАГОРОВА - СМИРНОВА =====================")
# Посчитаем D критическое для N=12200, alpha=0.1
___Dcrit = np.sqrt(- (np.log(0.5 * 0.1) / (2 * numOfPoints))) - 1 / (6 * numOfPoints)
print("\tКритическое значение = " + str(___Dcrit))

print("\tДля распределения Рэлея")
___D = 0
index = 1
for val in data:
    _____ddd = abs(stats.rayleigh.cdf(val, scale=sigma_for_rayleigh_moment_method) - index / numOfPoints)
    if _____ddd > ___D: ___D = _____ddd
    index += 1
print("\t\tДля метода моментов = " + str(___D))

___D = 0
index = 1
for val in data:
    _____ddd = abs(stats.rayleigh.cdf(val, scale=s_for_rayleigh_mmp) - index / numOfPoints)
    if _____ddd > ___D: ___D = _____ddd
    index += 1
print("\t\tДля ММП = " + str(___D))

print("\tДля распределения Лапласа")
___D = 0
index = 1
for val in data:
    _____ddd = abs(stats.laplace.cdf(val, loc=a_for_laplace_moment_method,
                                     scale=1 / laplace_lambda_moment_method) - index / numOfPoints)
    if _____ddd > ___D: ___D = _____ddd
    index += 1
print("\t\tДля метода моментов = " + str(___D))

___D = 0
index = 1
for val in data:
    _____ddd = abs(stats.laplace.cdf(val, loc=a_for_laplace_mmp, scale=1 / laplace_lambda_mmp) - index / numOfPoints)
    if _____ddd > ___D: ___D = _____ddd
    index += 1
print("\t\tДля ММП = " + str(___D))

print("\tДля Гамма-распределения")
___D = 0
index = 1
for val in data:
    _____ddd = abs(stats.gamma.cdf(val, k_for_gamma_moment_method,
                                   scale=theta_for_gamma_moment_method) - index / numOfPoints)
    if _____ddd > ___D: ___D = _____ddd
    index += 1
print("\t\tДля метода моментов = " + str(___D))

___D = 0
index = 1
for val in data:
    _____ddd = abs(stats.gamma.cdf(val, k_for_gamma_mmp, scale=theta_for_gamma_mmp) - index / numOfPoints)
    if _____ddd > ___D: ___D = _____ddd
    index += 1
print("\t\tДля ММП = " + str(___D))

# ======================= критерий Мизеса ================================
print("=============== статистика Мизеса =====================")
print("\tКритическое значение = 0.347")  # Значение взято из таблицы

print("\tДля распределения Рэлея")
___w = 0
index = 1
for val in data:
    ___w += (stats.rayleigh.cdf(val, scale=sigma_for_rayleigh_moment_method) - (2 * index - 1) / (2 * numOfPoints)) ** 2
    index += 1
___w = 1 / (12 * numOfPoints) + ___w
print("\t\tДля метода моментов = " + str(___w))

___w = 0
index = 1
for val in data:
    ___w += (stats.norm.cdf(val, scale=s_for_rayleigh_mmp) - (2 * index - 1) / (
            2 * numOfPoints)) ** 2
    index += 1
___w = 1 / (12 * numOfPoints) + ___w
print("\t\tДля ММП = " + str(___w))

print("\tДля распределения Лапласа")
___w = 0
index = 1
for val in data:
    ___w += (stats.laplace.cdf(val, loc=a_for_laplace_moment_method,
                               scale=1 / laplace_lambda_moment_method) - (2 * index - 1) / (2 * numOfPoints)) ** 2
    index += 1
___w = 1 / (12 * numOfPoints) + ___w
print("\t\tДля метода моментов = " + str(___w))

___w = 0
index = 1
for val in data:
    ___w += (stats.laplace.cdf(val, loc=a_for_laplace_mmp,
                               scale=1 / laplace_lambda_mmp) - (2 * index - 1) / (2 * numOfPoints)) ** 2
    index += 1
___w = 1 / (12 * numOfPoints) + ___w
print("\t\tДля ММП = " + str(___w))

print("\tДля Гамма-распределения")
___w = 0
index = 1
for val in data:
    ___w += (stats.gamma.cdf(val, k_for_gamma_moment_method,
                             scale=theta_for_gamma_moment_method) - (2 * index - 1) / (2 * numOfPoints)) ** 2
    index += 1
___w = 1 / (12 * numOfPoints) + ___w
print("\t\tДля метода моментов = " + str(___w))

___w = 0
index = 1
for val in data:
    ___w += (stats.gamma.cdf(val, k_for_gamma_mmp, scale=theta_for_gamma_mmp) - (2 * index - 1) / (
            2 * numOfPoints)) ** 2
    index += 1
___w = 1 / (12 * numOfPoints) + ___w
print("\t\tДля ММП = " + str(___w))
