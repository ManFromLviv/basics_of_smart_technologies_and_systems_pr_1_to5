import os
import sys
from io import StringIO

from ipywidgets import widgets

import src.split_data
import src.train_models
import src.predict_models

import tkinter as tk

def run():
    output_stream = StringIO()  # Створення потоку виводу з консолі
    sys.stdout = output_stream    # Перенаправлення потоку stdout
    try:
        text_area.delete(1.0, tk.END) # Очистка поля виводу інформації
        file_path = entry.get() # Отримання шляху з поля вводу
        src.split_data.split_data(file_path)
        n_estimators = n_estimators_slider.get()
        max_depth = max_depth_slider.get()
        max_iter_lr = max_iter_lr_slider.get()
        max_iter_svc = max_iter_svc_slider.get()
        src.train_models.train_models(n_estimators, max_depth, max_iter_lr, max_iter_svc)
        src.predict_models.predict_models()
        output_text = output_stream.getvalue()  # Отримання тексту для виводу інформації
        text_area.insert(tk.END, output_text)   # Вставка тексту в поле виводу інформації
    except Exception as e:
        text_area.insert(tk.END, "При вводі шляху до даних виникла помилка: " + str(e)) # Вставка тексту в поле виводу інформації

def open_data():
    os.startfile('data')

def open_source():
    os.startfile('src')

if __name__ == '__main__':
    # Створення вікна
    root = tk.Tk()

    # Отримання розмірів екрану
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Налаштування розміру і розшташування екрану відносно центру
    window_width = 700
    window_height = 700
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)

    # Налаштування вікна
    root.title("Практична робота № 5, Вальчевський П., ОІ-21сп")
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")  # Ширина, висота, позиція

    # Мітка
    label = tk.Label(root, text="Правильний шлях до даних:")
    label.pack(pady=1)  # Відступ

    # Поле вводу
    entry = tk.Entry(root, width=30)
    entry.pack(pady=1)  # Відступ
    entry.insert(0, "data/variant_2_new.csv") # Вставка дефолтного шляху до даних

    # Кнопка Почати!
    button_run = tk.Button(root, text="Почати!", command=run, width=40)
    button_run.pack(pady=1)  # Відступ

    # Кнопка Відкрити папку з даними
    button_open_data = tk.Button(root, text="Відкрити папку з даними", command=open_data, width=40)
    button_open_data.pack(pady=1)  # Відступ

    # Кнопка Відкрити папку з моделями
    button_open_source = tk.Button(root, text="Відкрити папку з ресурсами", command=open_source, width=40)
    button_open_source.pack(pady=1)  # Відступ

    # Мітка і повзунки для гіперпараметрів RandomForest
    label_rf = tk.Label(root, text="Налаштування Random Forest:")
    label_rf.pack(pady=1)  # Відступ
    n_estimators_slider = tk.Scale(root, from_=2, to=25, orient=tk.HORIZONTAL, label="кількість дерев")
    n_estimators_slider.set(2)
    n_estimators_slider.pack(pady=1)
    max_depth_slider = tk.Scale(root, from_=5, to=20, orient=tk.HORIZONTAL, label="макс. глибина")
    max_depth_slider.set(5)
    max_depth_slider.pack(pady=1)

    # Мітка і повзунки для гіперпараметрів Logistic Regression
    label_lr = tk.Label(root, text="Налаштування Logistic Regression:")
    label_lr.pack(pady=1)  # Відступ
    max_iter_lr_slider = tk.Scale(root, from_=10, to=400, orient=tk.HORIZONTAL, label="кількість ітер.")
    max_iter_lr_slider.set(10)
    max_iter_lr_slider.pack(pady=1)

    # Мітка і повзунки для гіперпараметрів SVC
    label_svc = tk.Label(root, text="Налаштування SVC:")
    label_svc.pack(pady=1)  # Відступ
    max_iter_svc_slider = tk.Scale(root, from_=2, to=100, orient=tk.HORIZONTAL, label="кількість ітер.")
    max_iter_svc_slider.set(2)
    max_iter_svc_slider.pack(pady=1)

    # Вивід інформації
    text_area = tk.Text(root, height=500, width=500)
    text_area.pack(pady=5)  # Відступ

    # Запуск
    root.mainloop()