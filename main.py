import os
import sys
from io import StringIO

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
        src.train_models.train_models()
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
    label.pack(pady=5)  # Відступ

    # Поле вводу
    entry = tk.Entry(root, width=30)
    entry.pack(pady=5)  # Відступ
    entry.insert(0, "data/variant_2_new.csv") # Вставка дефолтного шляху до даних

    # Кнопка Почати!
    button_run = tk.Button(root, text="Почати!", command=run, width=40)
    button_run.pack(pady=5)  # Відступ

    # Кнопка Відкрити папку з даними
    button_open_data = tk.Button(root, text="Відкрити папку з даними", command=open_data, width=40)
    button_open_data.pack(pady=5)  # Відступ

    # Кнопка Відкрити папку з моделями
    button_open_source = tk.Button(root, text="Відкрити папку з ресурсами", command=open_source, width=40)
    button_open_source.pack(pady=5)  # Відступ

    # Вивід інформації
    text_area = tk.Text(root, height=500, width=500)
    text_area.pack(pady=5)  # Відступ

    # Запуск
    root.mainloop()