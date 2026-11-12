import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
"""

try:
    df = pd.read_csv("C:/german_used_cars.csv")
    print("Файл прочитан успешно")
except FileNotFoundError:
    print("файл не найден")
except Exception as e:
    print("иная ошибка e")
"""

#читаею файл и вывожу значения для проверки
df = pd.read_csv("C:/german_used_cars.csv")

#основная информация
#print(df.info())
#print("\n",df.describe())


#Проведите оценку данных по столбцам, посмотрите визуально на распределение данных,
# определите базовые статистики, где это возможно
#определение базовых статистик
#df['mileage_in_km'] = pd.to_numeric(df['mileage_in_km'], errors='coerce')
#print(df[(df["mileage_in_km"] > 1) & (df["mileage_in_km"] < 3)]["mileage_in_km"].median())

#ассчитаем милимитраж по модели
grouped_df = df.groupby("model")["mileage_in_km"].median().sort_values()
#print(grouped_df)
"""с помощью метода transform можем увидеть взаимосвязь
grouped = df["mileage_in_km"].fillna(df.groupby("model")["mileage_in_km"].transform("median"))
print(grouped)
"""
#Найдите выбросы в данных, для наглядности используйте соответствующие типы визуализаций.
#попробуйте объяснить их появление в наборе

#меняю строковые значения на числовые
df['power_kw'].apply(type).value_counts()
df["power_kw"] = pd.to_numeric(df["power_kw"],errors="coerce")
#print(df["power_kw"].dtypes)
grouped = df["power_kw"].fillna(df.groupby("model")["power_kw"].transform("median"))

#ыбросы
q1 = df["power_kw"].quantile(0.25)
q3 = df["power_kw"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - (iqr * 1.5)
upper_bound = q3 + (iqr * 1.5)
remove_outliers = df[df["power_kw"].between(lower_bound,upper_bound,inclusive="both")]
#print(remove_outliers)


num_bins = 10
df['power_kw_group'] = pd.cut(df["power_kw"],bins=num_bins,labels=False, include_lowest=True)
plt.figure(figsize=(14, 7))
ax = sns.countplot(data=df, y="power_kw_group", order=df['power_kw_group'].value_counts().index)
ax.bar_label(ax.containers[0])
plt.title("Количество машин по группам мощности", fontsize=16)
plt.xlabel("Количество машин", fontsize=12)
plt.ylabel("Группа мощности (kW)", fontsize=12)
plt.xticks(rotation=45, ha='right') # Поворот подписей оси X, если они накладываются
plt.tight_layout() # Автоматически подбирает параметры, чтобы элементы графика не накладывались
plt.show()