import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from category_encoders.binary import BinaryEncoder
from sklearn.preprocessing import StandardScaler

from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

# Загрузка датасета
data = pd.read_csv("online_shoppers_intention.csv")

# Разделение данных на фичи и лэйблы;
# Разделение фич на численные и категориальные параметры;
# Отбрасывание части категориальных параметров ввиду их низкого влияния на итоговый результат (тип браузера, тип трафика, тип операционной системы)
# Предобработка данных: нормирование и выравнивание матожидания для численных параметров и бинарное кодирование для категориальных.
X_num = StandardScaler().fit_transform(data[data.columns[0:10]])
X_cat = np.multiply(BinaryEncoder().fit_transform(data[['Month', 'VisitorType', 'Weekend']]).values, 1)

y = np.multiply(data['Revenue'].values, 1)

XTemp = []
for i in range(0, len(X_num)):
    XTemp.append(np.append(X_num[i], X_cat[i]))

X = np.asarray(XTemp)

# Разделение данных на обучающую и тестовую выборки, 20% на тестовую.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Выравнивание тренировочной выборки датасета.
# Был выбран андерсемплинг, так из-за малого объёма одной из категорий (порядка 20% от всей выборки),
# а также значительной перекрёстной связности данных качество генерация новых данных было не высоким и
# качество определения малой категории очень существенно снижалось
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X_train, y_train)

# Создание пайплайна с предварительной обработкой, выбором признаков и моделью GBM
# Модель была выбрана по причине достаточно большого количества входных параметров
# RFM показал себя практически аналогично, однако результаты в зависимости
# от обучающей выборки плавали в более широком диапозоне
pipeline = Pipeline([
    ('GBC', GradientBoostingClassifier())
])

# Определение сетки гиперпараметров для поиска
# Полученные мной ранее оптимальные гиперпараметры:
# GBC__learning_rate': 0.1, 'GBC__max_depth': 5, 'GBC__min_samples_leaf': 7, 'GBC__min_samples_split': 3, 'GBC__n_estimators': 50
param_grid = {
    'GBC__n_estimators': [25, 50, 75, 100],
    'GBC__learning_rate': [0.05, 0.1, 0.2],
    'GBC__max_depth': [3, 5, 7],
    'GBC__min_samples_split': [3, 5, 7, 9, 11],
    'GBC__min_samples_leaf': [3, 5, 7, 9, 11]
}

# Настраиваем поиск гиперпараметров
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=2
)

# Обучение модели с поиском гиперпараметров
grid_search.fit(X_rus, y_rus)

# Вывод лучших параметров
print("Лучшие параметры:", grid_search.best_params_)

# Предсказание на тестовом наборе с использованием лучшей модели
y_pred = grid_search.predict(X_test)

# Метрики качества
print("Метрики качества:")
print(classification_report(y_test, y_pred))

# Матрица ошибок
conf_matrix = confusion_matrix(y_test, y_pred)
print("Матрица ошибок:")
print(conf_matrix)

# Точность модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.4f}")

# Визуализация матрицы ошибок
import seaborn as sns

plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Класс 0', 'Класс 1'], yticklabels=['Класс 0', 'Класс 1'])
plt.xlabel('Предсказанные классы')
plt.ylabel('Истинные классы')
plt.title('Матрица ошибок')
plt.show()

# В итоге была достигнута точность порядка 85%. Основной проблемой является достаточно сильно разбалансированный датасет.