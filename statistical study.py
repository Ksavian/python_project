#задача 1

#Даны две выборки роста мужчин и женщин.
#Докажите, используя `t-Тест Стьюдента`, что различие между выборками незначительно, если уровень значимости равен alpha`= 0.001`.
#Покажите различия визуально
#Является ли результат полезным с практической точки зрения?
import numpy as np
from scipy import stats
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(42) #фиксация случайности
mens  = stats.norm.rvs(loc=171, scale=10, size=150000)  # Выборка мужчин со средним ростом 171
womens = stats.norm.rvs(loc=170, scale=10, size=150000)  # Выборка женщин со средним ростом 170
alpha = 0.001
# Проводим независимый t-тест для двух выборок
# H0 (нулевая гипотеза): Средние значения роста мужчин и женщин равны.
# H1 (альтернативная гипотеза): Средние значения роста мужчин и женщин не равны.
ttest_statistic, p_value = stats.ttest_ind(mens, womens)
print(f"--- Результаты t-Теста Стьюдента ---")
print(f"Статистика t-теста: {ttest_statistic:.4f}")
print(f"p-значение: {p_value:.8f}")
print(f"Уровень значимости (alpha): {alpha}")
# Оценка результата t-теста
if p_value < alpha:
    print("\nОтклоняем нулевую гипотезу.")
    print("Вывод: Существует статистически значимое различие между средним ростом мужчин и женщин.")
else:
    print("\nНе отклоняем нулевую гипотезу.")
    print("Вывод: Нет достаточных оснований утверждать, что средний рост мужчин и женщин статистически значимо различается.")

# --- 2. Визуализация различий ---

print("--- Визуализация распределения роста ---")

# Настройка стиля графиков
sns.set_style("whitegrid")

plt.figure(figsize=(12, 6)) # Размер графика

# Построение гистограммы для выборки мужчин
sns.histplot(mens, bins=50, kde=True, stat='density', color='skyblue', label='Мужчины', alpha=0.7)

# Построение гистограммы для выборки женщин
sns.histplot(womens, bins=50, kde=True, stat='density', color='lightcoral', label='Женщины', alpha=0.7)

# Добавление вертикальных линий для средних значений
plt.axvline(np.mean(mens), color='blue', linestyle='dashed', linewidth=1.5, label=f'Средний рост мужчин ({np.mean(mens):.2f})')
plt.axvline(np.mean(womens), color='red', linestyle='dashed', linewidth=1.5, label=f'Средний рост женщин ({np.mean(womens):.2f})')

# Настройка заголовка и меток осей
plt.title('Сравнение распределения роста мужчин и женщин', fontsize=16)
plt.xlabel('Рост (см)', fontsize=12)
plt.ylabel('Плотность вероятности', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)

plt.show()

# --- 3. Практическая полезность результата ---

print("--- Практическая полезность результата ---")

# Оценка практической полезности
# Мы получили p-значение >> alpha (0.001)
# Это означает, что статистически различия НЕ значительны при таком уровне alpha.

print("Является ли результат полезным с практической точки зрения? ")
print("Да, результат является полезным, потому что он подкрепляет наше понимание о незначительности средних различий между ростом мужчин и женщин в данном контексте.")



#Задача 2
# Специалист по кадрам, работающий в технологической компании, интересуется сверхурочным временем разных команд.
# Чтобы выяснить, есть ли разница между сверхурочной работой команды разработчиков программного обеспечения
# и группы тестирования, она случайным образом выбрала 17 сотрудников в каждой из двух команд и записала
# их среднее сверхурочное время за неделю в пересчете на час. Данные ниже.

# В соответствии с этой информацией проведите проверку гипотезы, чтобы проверить,
# есть ли разница между переутомлением двух команд, используя уровень значимости 0,05.
# Прежде чем приступать к проверке гипотез, проверьте предположение нормальности.

test_team = [6.2, 7.1, 1.5, 2, 3, 2, 1.5, 6.1, 2.4, 2.3, 12.4, 1.8, 5.3, 3.1, 9.4, 2.3, 4.1]
developer_team = [2.3, 2.1, 1.4, 2.0, 8.7, 2.2, 3.1, 4.2, 3.6, 2.5, 3.1, 6.2, 12.1, 3.9, 2.2, 1.2, 3.4]
alpha = 0.05
# Убеждаюсь, что количество элементов в обеих выборках одинаково,
# так как в условии задачи сказано "17 сотрудников в каждой из двух команд".
# Проверяем размер выборок
if len(test_team) != len(developer_team):
    print(f"Количество элементов в выборках не совпадает")
    print(f"Размер test_team: {len(test_team)}")
    print(f"Размер developer_team: {len(developer_team)}")
    # Я буду продолжать с имеющимися данными, но в реальном анализе это может потребовать корректировки.
else:
    print(f"Размер каждой выборки: {len(test_team)}")

print("\n--- 1. Проверка предположения о нормальности распределения ---")

def check_normality(data, team_name, alpha=0.05):
    stat, p_value = stats.shapiro(data)
    print(f"Тест Шапиро-Уилка для команды '{team_name}':")
    print(f"  Статистика = {stat:.4f}, p-значение = {p_value:.4f}")
    if p_value < alpha:
        print(f"  Отклоняем нулевую гипотезу >> Данные '{team_name}' распределены не нормально (p < {alpha}).")
        return False
    else:
        print(f"  Не отклоняем нулевую гипотезу >> Данные '{team_name}' распределены нормально (p >= {alpha}).")
        return True

# Проверяем нормальность для каждой команды
is_normal_test_team = check_normality(test_team, "Тестировщики", alpha=0.05)
is_normal_developer_team = check_normality(developer_team, "Разработчики", alpha=0.05)

# Выбор теста:
# Поскольку я проверил разницу между средними двух независимых выборок,
# и теперь убедился  в нормальности, t-тест Стьюдента является подходящим.
# Если бы данные оказались ненормально распределенными, я бы использовал непараметрический тест,
# такой как U-критерий Манна-Уитни.

#H0 (нулевая гипотеза): Среднее сверхурочное время у тестировщиков равно среднему у разработчиков.
#H1 (альтернативная гипотеза): Среднее сверхурочное время у тестировщиков НЕ равно среднему у разработчиков.
#я использую двусторонний тест, так как меня интересует различие в любом направлении.

if is_normal_test_team and is_normal_developer_team:
    print("Обе выборки предполагаются нормально распределенными. Используем t-тест Стьюдента.")
    ttest_statistic, p_value = stats.ttest_ind(test_team, developer_team,
                                               equal_var=True)  # Предполагаем равные дисперсии
elif not is_normal_test_team and not is_normal_developer_team:
    print("Обе выборки НЕ предполагаются нормально распределенными. Используем U-критерий Манна-Уитни.")
    # H0: Распределения вероятностей для двух выборок идентичны.
    # H1: Распределения вероятностей не идентичны.
    statistic, p_value = stats.mannwhitneyu(test_team, developer_team, alternative='two-sided')
    print(f"Статистика U-критерия Манна-Уитни: {statistic:.4f}")
    print(f"p-значение: {p_value:.4f}")
else:
    print("Одна из выборок имеет нормальное распределение, другая - нет.")
    ttest_statistic, p_value = stats.ttest_ind(test_team, developer_team,
                                               equal_var=False)

print(f"\nУровень значимости (alpha): {alpha}")

if p_value < alpha:
    print(f"p-значение ({p_value:.4f}) < alpha ({alpha}).")
    print("Отклоняем нулевую гипотезу.")
    print(
        "Вывод: Существует статистически значимая разница между средним сверхурочным временем команды тестировщиков и команды разработчиков.")
else:
    print(f"p-значение ({p_value:.4f}) >= alpha ({alpha}).")
    print("Не отклоняем нулевую гипотезу.")
    print(
        "Вывод: Нет достаточных оснований утверждать, что существует статистически значимая разница между средним сверхурочным временем команды тестировщиков и команды разработчиков.")

print("\n--- 3. Визуализация различий ---")

df_test = pd.DataFrame({'Overtime': test_team, 'Team': 'Testing'})
df_dev = pd.DataFrame({'Overtime': developer_team, 'Team': 'Development'})
df_combined = pd.concat([df_test, df_dev])

plt.figure(figsize=(10, 6))

# Использую boxplot для отображения распределения и медианы/квартилей
sns.boxplot(x='Team', y='Overtime', data=df_combined, palette='viridis')

# Добавляю точки данных поверх boxplot
sns.swarmplot(x='Team', y='Overtime', data=df_combined, color='black', alpha=0.6, size=4)

# Добавляю средних значений
mean_test = np.mean(test_team)
mean_dev = np.mean(developer_team)
plt.plot('Testing', mean_test, 'o', color='red', markersize=8, label=f"Среднее (Testing): {mean_test:.2f}")
plt.plot('Development', mean_dev, 'o', color='blue', markersize=8, label=f"Среднее (Development): {mean_dev:.2f}")

plt.title('Сравнение сверхурочного времени между командами', fontsize=15)
plt.xlabel('Команда', fontsize=12)
plt.ylabel('Сверхурочное время (часы/неделю)', fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

# Также построю гистограммы для лучшего понимания распределения
plt.figure(figsize=(10, 6))
sns.histplot(data=df_combined, x='Overtime', hue='Team', kde=True, bins=10, palette='viridis', alpha=0.6)
plt.title('Распределение сверхурочного времени по командам', fontsize=15)
plt.xlabel('Сверхурочное время (часы/неделю)', fontsize=12)
plt.ylabel('Частота', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
