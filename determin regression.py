from regression_analyz import X_test, y_train
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

cmap = colors.LinearSegmentedColormap(
    'red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)
wine = load_wine()
print(wine.DESCR)

# загружаю данные в датафрейм
data = pd.DataFrame(wine.data,columns=wine.feature_names)
data_class = wine.target

# разделяю выборку на тренировочную и тестовую
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(data,data_class,test_size=0.25, random_state=42)
len(y_test)

# импортирую LDA-функционал
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()

# обучаю данные
lda.fit(X_train, y_train)

# делаю прогноз на тестовой выборке
lda.predict(X_test)

# смотрю разницу факта и прогноза
result = pd.DataFrame([y_test, lda.predict(X_test)]).T

from sklearn.metrics import accuracy_score
accuracy_score(y_test, lda.predict(X_test))

# коэффициенты дискриминатных линий
print(lda.coef_)

# визуализирую дискриминантную линию только по 2 признакам
plt.scatter(x=X_train['ash'], y=X_train['flavanoids'], c=y_train)
plt.show()

#### Кластеризация на этом же наборе данных с этой же моделью
small_train = X_train[(y_train == 0) | (y_train == 2)]
small_train = small_train[['ash', 'flavanoids']]
sy_train = y_train[(y_train == 0) | (y_train == 2)]
small_test = X_test[(y_test == 0) | (y_test == 2)]
small_test = small_test[['ash', 'flavanoids']]
sy_test = y_test[(y_test == 0) | (y_test == 2)]

plt.scatter(x=small_train['ash'], y=small_train['flavanoids'], c=sy_train)
plt.show()
lda_small = LinearDiscriminantAnalysis()
lda_small.fit(small_train, sy_train)

# посмотрю на наши центроиды
print(lda_small.means_)


plt.scatter(small_train['ash'], small_train['flavanoids'], c=sy_train)
# центроиды
plt.scatter(lda_small.means_[:, 0], lda_small.means_[:, 1], c='r', s=150, marker='*')

# делаю условную "сетку"
nx, ny = 200, 100
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))

# предсказываю класс каждой точки нашей сетки
Z = lda_small.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z[:, 1].reshape(xx.shape)

# закрашиваю классы разными цветами
plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
                   norm=colors.Normalize(0., 1.), zorder=-1, shading='auto')

plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='white')

# второй признак во много раз важнее первого для кластеризации (это видно по визуализации)
print(lda_small.coef_)

# какая будет точность?
accuracy_score(sy_test, lda_small.predict(small_test))
data[['flavanoids', 'ash']].corr()

# заменю модель на логистическую регрессиию

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(small_train, sy_train)
lr.predict(small_test)
lr.predict_proba(small_test)

accuracy_score(sy_test, lr.predict(small_test))
print(lr.coef_)

plt.scatter(small_train['ash'], small_train['flavanoids'], c=sy_train)

nx, ny = 200, 100
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
Z = lr.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z[:, 1].reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
                   norm=colors.Normalize(0., 1.), zorder=-1, shading='auto')
plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='white')

plt.show()

### Кластеризация
# генерирую искуссвтенные данные для примера
from sklearn.datasets import make_blobs

# 200 точек с 2 центрами, СКО - 2.5
X, y = make_blobs(n_samples=200, centers=2, cluster_std=2.5)

plt.scatter(X[:, 0], X[:, 1], cmap='autumn', s=60)

# приведу данные к единому масштабу
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

v_1 = np.array(np.random.randint(0, 1000, 500))
v_2 = np.array(np.random.randint(0, 10, 500))
print(v_1.max())
print(v_1.min())

print(v_2.max())
print(v_2.min())

v_1_scaled = scaler.fit_transform(v_1.reshape(-1, 1))
v_2_scaled = scaler.fit_transform(v_2.reshape(-1, 1))
X_scaled = scaler.fit_transform(X)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1],cmap='autumn', s=60)

from sklearn.cluster import KMeans
# явно указываю количество кластеров
kmeans = KMeans(n_clusters=2)
# fit_predict обучается на данных и каждому объекту присваивает кластер
clusters = kmeans.fit_predict(X_scaled)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], cmap='autumn', c=clusters, s=60)

# строю график локтя
# создаю список для инерции
k_inertia = []
# задаю диапазон кластеров
ks = range(1, 20)

for k in ks:
    clf_kmeans = KMeans(n_clusters=k)
    clusters_kmeans = clf_kmeans.fit_predict(X_scaled)
    # добавляю инерцию каждой модели в список
    k_inertia.append(clf_kmeans.inertia_)
plt.plot(ks, k_inertia)
plt.plot(ks, k_inertia ,'ro')

from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
# строю дендрограмму
distance_mat = pdist(X_scaled) # pdist посчитает нам матрицу попарных расстояний между точками
# Z = hierarchy.linkage(distance_mat) # linkage — реализация агломеративного алгоритма
Z = hierarchy.linkage(distance_mat, 'ward')
plt.figure(figsize=(20, 9))
dn = hierarchy.dendrogram(Z)

# делаею уже кластеризацию по факту
from sklearn.cluster import AgglomerativeClustering
aglo = AgglomerativeClustering(n_clusters=2)
clusters = aglo.fit_predict(X_scaled)


plt.scatter(X_scaled[:, 0], X_scaled[:, 1],cmap='autumn', c=clusters, s=60)
df = pd.DataFrame(X)
df['y_hat'] = clusters5


