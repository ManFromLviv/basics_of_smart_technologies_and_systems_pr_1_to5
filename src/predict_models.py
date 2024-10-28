#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import joblib
from datetime import datetime
from sklearn.metrics import classification_report

def predict_models():
    # Функція виводу метрик та збереження їх у файл
    def classification_metrics(y_test, y_pred, name=str, file_path=str):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('Звіт метрик для: ' + name + '\n')
            if y_test is not None:  # Перевірка наявності цільової змінної
                report = classification_report(y_test, y_pred)
                f.write(report)
            else:
                f.write("Цільова змінна 'Species' відсутня, метрики не можуть бути обчислені.")
        with open(file_path, 'r', encoding='utf-8') as f:
            print(f.read())

    # Функція для передбачення моделі
    def predict_model(model, name_model=str):
        # Завантаження даних для передбачення
        new_input = pd.read_csv('data/new_input.csv')
        species_true = new_input['Species'] # Збереження значень цільової змінної
        if 'Species' in new_input.columns: # Видалення цільової змінної
            new_input = new_input.drop('Species', axis=1)

        # Передбачення моделлю
        predictions = model.predict(new_input)

        # Збереження передбачень (наприклад: random_search_log_reg_predictions_2024-10-28_02-40-31.csv або .txt)
        str_datetime_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name_file = 'data/predictions/' + name_model + '_predictions_' + str_datetime_now + '.csv'
        name_file_report = 'data/predictions/' + name_model + '_predictions_' + str_datetime_now + '.txt'
        pd.DataFrame(predictions, columns=['Predictions']).to_csv(name_file, index=False)
        print("Файл передбачення було збережено та звіт метрик до нього також.")
        classification_metrics(species_true, predictions, name_file, name_file_report)

    # Завантаження моделі
    predict_model(joblib.load('src/models/random_search_rf.pkl'), 'random_search_rf')
    predict_model(joblib.load('src/models/random_search_log_reg.pkl'), 'random_search_log_reg')
    predict_model(joblib.load('src/models/random_search_svm.pkl'), 'random_search_svm')


