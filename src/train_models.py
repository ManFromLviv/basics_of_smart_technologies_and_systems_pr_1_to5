#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from scipy.stats import randint
import joblib

def train_models(n_estimators, max_depth, max_iter_lr, max_iter_svc):
    # Завантаження тренувальних даних
    train_data = pd.read_csv('data/train_split.csv')

    # Кодування категоріальних стовпців
    cat_columns = ['Date Egg', 'Clutch Completion', 'studyName', 'Region', 'Island', 'Stage', 'Individual ID', 'Sex', 'Species', 'Comments']
    map_dicts = dict()
    for column in cat_columns:
        train_data[column] = train_data[column].astype('category')  # Перетворення на категоріальний тип
        map_dicts[column] = dict(enumerate(train_data[column].cat.categories))  # Збереження кодування для зворотного перетворення
        train_data[column] = train_data[column].cat.codes  # Заміна значень на числові коди


    # Розподіл даних на ознаки і цільову змінну
    X_train = train_data.drop('Species', axis=1)
    y_train = train_data['Species']

    # Ініціалізація та налаштування моделі RandomForestClassifier
    def use_RandomForestClassifier(n_estimators, max_depth):
        rf = RandomForestClassifier(random_state=42)
        n_estimators = list(range(1, n_estimators))
        max_depth = list(range(4, max_depth))
        # Гіперпараметри
        param_distributions_rf = {
            'n_estimators': n_estimators,  # Кількість дерев
            'max_depth': max_depth,  # Максимальна глибина кожного дерева
            'min_samples_split': randint(2, 20),  # Мінімальна кількість зразків для поділу вузла
            'min_samples_leaf': randint(2, 20),  # Мінімальна кількість зразків у листі
            'max_features': [0.1]  # Кількість ознак у %, що використовуються при кожному розгалуженні
        }

        # Оптимізація гіперпараметрів
        random_search_rf = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_distributions_rf,
            n_iter=100,  # Кількість ітерацій
            scoring='f1',  # Метрика оцінювання (F1 score)
            cv=5,  # Поділ даних на 5 частин (4 тренуються, 1 тестуються)
            n_jobs=-1,  # Використання всіх доступних ядер процесора
            random_state=42
        )

        # Навчання моделі
        random_search_rf.fit(X_train, y_train)

        # Збереження моделі
        joblib.dump(random_search_rf, 'src/models/random_search_rf.pkl')

        print('Модель RandomForestClassifier було налаштовано, навчено та збережено!')

    # Ініціалізація та налаштування моделі LogisticRegression
    def use_LogisticRegression(max_iter_lr):
        log_reg = LogisticRegression(random_state=42)
        max_iter_lr = randint(1, max_iter_lr)
        # Гіперпараметри
        param_distributions_log_reg = {
            'C': uniform(0.1, 5),  # Рівень регуляризації
            'penalty': ['l1', 'l2', 'elasticnet'],  # Тип регуляризації (долучення втрат коефіцієнтів та їх квадратів)
            'solver': ['liblinear', 'saga', 'lbfgs'],  # Алгоритм для розв'язання (лінійний, градієнтний спуск, обмеженої пам'яті)
            'l1_ratio': uniform(0, 1),  # Значення для elasticnet, от 0 до 1
            'max_iter': max_iter_lr,  # Кількість ітерацій
            'tol': uniform(1e-6, 1e-2)  # Допустима похибка
        }

        # Оптимізація гіперпараметрів
        random_search_log_reg = RandomizedSearchCV(
            estimator=log_reg,
            param_distributions=param_distributions_log_reg,
            scoring='f1',  # Метрика оцінювання (F1 score)
            cv=5,  # Поділ даних на 5 частин (4 тренуються, 1 тестуються)
            n_jobs=-1,  # Використання всіх доступних ядер процесора
            random_state=400
        )

        # Навчання моделі
        random_search_log_reg.fit(X_train, y_train)

        # Збереження моделі
        joblib.dump(random_search_log_reg, 'src/models/random_search_log_reg.pkl')

        print('Модель LogisticRegression було налаштовано, навчено та збережено!')

    # Ініціалізація та налаштування моделі SVC
    def use_SVC(max_iter_svc):
        svm_model = SVC(random_state=42)
        max_iter_svc = randint(1, max_iter_svc)
        # Гіперпараметри
        param_distributions_svm = {
            'C': uniform(0.01, 10),  # Рівень регуляризації
            'kernel': ['linear', 'rbf'],  # Ядро SVM (лінійне, радіально базисна функція)
            'gamma': uniform(0.01, 1),  # Числові значення для gamma (вплив окремих зразків на результат для радіально базисної функції)
            'max_iter':max_iter_svc  # Кількість ітерацій
        }

        # Оптимізація гіперпараметрів
        random_search_svm = RandomizedSearchCV(
            estimator=svm_model,
            param_distributions=param_distributions_svm,
            n_iter=100,  # Кількість ітерацій
            scoring='f1',  # Метрика оцінювання (F1 score)
            cv=5,  # Поділ даних на 5 частин (4 тренуються, 1 тестуються)
            n_jobs=-1,  # Використання всіх доступних ядер процесора
            random_state=42
        )

        # Навчання моделі
        random_search_svm.fit(X_train, y_train)

        # Збереження моделі
        joblib.dump(random_search_svm, 'src/models/random_search_svm.pkl')

        print('Модель SVC було налаштовано, навчено та збережено!')

    # Використання моделей
    use_RandomForestClassifier(n_estimators, max_depth)
    use_LogisticRegression(max_iter_lr)
    use_SVC(max_iter_svc)