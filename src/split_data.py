#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(file_path):
    # Завантаження даних (які були оброблені мною у одній з ПР)
    data = pd.read_csv(file_path)

    # Категоріальні стовпці
    cat_columns = ['Date Egg', 'Clutch Completion', 'studyName', 'Region', 'Island', 'Stage', 'Individual ID', 'Sex', 'Species', 'Comments']

    # Кодування категоріальних стовпців
    map_dicts = dict()
    for column in cat_columns:
        data[column] = data[column].astype('category')
        map_dicts[column] = dict(zip(data[column], data[column].cat.codes))
        data[column] = data[column].cat.codes

    # Розділення на train та new_input у співвідношенні 90:10
    train_data, new_input = train_test_split(data, test_size=0.1, random_state=42)

    # Збереження даних
    train_data.to_csv('data/train_split.csv', index=False)
    new_input.to_csv('data/new_input.csv', index=False)
    print('Дані були закодовані, розділені та збережені!')

