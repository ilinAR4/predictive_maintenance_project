# Проект: Бинарная классификация для предиктивного обслуживания оборудования

## Описание проекта

Цель проекта — разработать модель машинного обучения, которая предсказывает, произойдет ли отказ оборудования (Target = 1) или нет (Target = 0). Результаты работы оформлены в виде многостраничного Streamlit-приложения с использованием навигации.

Проект предназначен для демонстрации возможностей машинного обучения в области предиктивного обслуживания, что позволяет планировать техническое обслуживание оборудования только при необходимости, а не по фиксированному графику.

## Датасет

Используется датасет **"AI4I 2020 Predictive Maintenance Dataset"**, содержащий 10 000 записей с 14 признаками. Подробное описание датасета можно найти в [документации](https://archive.ics.uci.edu/dataset/601/predictive+maintenance+dataset).

Основные характеристики данных:

- **Переменные (Features)**:
  - UDI: Уникальный идентификатор записи
  - Product ID: Идентификатор продукта (L, M, H) и серийный номер
  - Type: Тип продукта (L, M, H)
  - Air temperature [K]: Температура окружающей среды
  - Process temperature [K]: Рабочая температура процесса
  - Rotational speed [rpm]: Скорость вращения
  - Torque [Nm]: Крутящий момент
  - Tool wear [min]: Износ инструмента

- **Целевые переменные (Targets)**:
  - Machine failure: Бинарная метка (0 - нет отказа, 1 - отказ)
  - TWF: Отказ из-за износа инструмента
  - HDF: Отказ из-за недостаточного теплоотвода
  - PWF: Отказ из-за недостаточной или избыточной мощности
  - OSF: Отказ из-за перегрузки
  - RNF: Случайный отказ

## Структура репозитория
predictive_maintenance_project/  
├── app.py                              # Основной файл Streamlit-приложения (entrypoint)  
├── analysis_and_model.py               # Основная страница с анализом данных и моделью  
├── presentation.py                     # Страница с презентацией проекта  
├── requirements.txt                    # Файл с зависимостями для установки библиотек  
├── data/                               # Папка с данными  
│   └── predictive_maintenance.csv      # Датасет  
├── models/                             # Папка для сохранения обученных моделей  
└── README.md                           # Описание проекта

## Установка и запуск

1. Клонируйте репозиторий:  
git clone https://github.com/ilinAR4/predictive_maintenance_project  
cd predictive_maintenance_project  

2. Установите зависимости:  
pip install -r requirements.txt  

3. Запустите приложение:  
streamlit run app.py  




## Функции приложения

Приложение состоит из двух основных страниц:

### 1. Анализ и модель

На этой странице пользователь может:
- Загрузить данные из файла или использовать библиотеку ucimlrepo для загрузки датасета
- Выполнить предобработку данных (удаление лишних столбцов, кодирование категориальных переменных, масштабирование)
- Обучить различные модели машинного обучения (Логистическая регрессия, Случайный лес, XGBoost, SVM)
- Оценить качество моделей с помощью метрик (Accuracy, Precision, Recall, F1, ROC-AUC)
- Визуализировать результаты (матрица ошибок, ROC-кривые, важность признаков)
- Сделать предсказание на новых данных

### 2. Презентация

На этой странице представлена информация о проекте в формате презентации, включая:
- Введение и описание задачи
- Описание датасета
- Методология решения
- Результаты анализа
- Выводы и возможные улучшения

## Предобработка данных

В проекте выполняются следующие шаги предобработки:

1. **Удаление ненужных столбцов**:
- Удаляются столбцы с уникальными идентификаторами (UDI, Product ID), которые не несут полезной информации для модели.
- Опционально удаляются дополнительные целевые переменные (TWF, HDF, PWF, OSF, RNF), так как основная задача - прогнозирование общего отказа (Machine failure).

2. **Преобразование категориальных переменных**:
- Категориальный признак Type преобразуется в числовой формат с помощью LabelEncoder, так как модели машинного обучения работают только с числами.

3. **Проверка на пропущенные значения**:
- Проверяется наличие пропущенных значений в данных.

4. **Масштабирование данных**:
- Числовые признаки масштабируются с помощью StandardScaler для улучшения сходимости моделей.

## Разделение данных

Данные разделяются на обучающую и тестовую выборки в соотношении 80/20. Это стандартное соотношение, которое обеспечивает достаточный объем данных как для обучения, так и для проверки моделей.

## Обучение модели

В проекте используются следующие модели машинного обучения:

1. **Логистическая регрессия**:
- Простая и интерпретируемая модель, подходящая для бинарной классификации.

2. **Случайный лес**:
- Устойчивая к переобучению модель, способная работать с нелинейными данными.

3. **XGBoost**:
- Мощная модель, которая часто показывает высокую точность на сложных данных.

4. **Support Vector Machine (SVM)**:
- Эффективна на данных с высокой размерностью, подходит для сложных границ решений.

## Оценка модели

Для оценки качества моделей используются следующие метрики:

- **Accuracy** (Точность): Доля правильных предсказаний.
- **Confusion Matrix** (Матрица ошибок): Таблица, показывающая количество правильных и неправильных предсказаний для каждого класса.
- **Precision** (Точность): Доля правильно предсказанных положительных примеров среди всех предсказанных положительных.
- **Recall** (Полнота): Доля правильно предсказанных положительных примеров среди всех положительных.
- **F1-score**: Гармоническое среднее между точностью и полнотой.
- **ROC-AUC**: Площадь под ROC-кривой, показывающая способность модели разделять классы.

## Streamlit-приложение

Приложение реализовано с использованием библиотеки Streamlit, которая позволяет быстро создавать интерактивные веб-приложения на Python.

Основные преимущества Streamlit:
- Простота использования
- Быстрая разработка
- Множество встроенных виджетов для взаимодействия с пользователем
- Поддержка визуализации данных

В приложении реализованы следующие функции:
- Загрузка и отображение данных
- Предобработка данных
- Обучение и оценка моделей
- Визуализация результатов
- Предсказание на новых данных

## Используемые библиотеки

- streamlit==1.41.0 - для создания веб-интерфейса
- pandas==1.5.3 - для работы с данными
- scikit-learn==1.2.0 - для создания моделей машинного обучения
- matplotlib==3.6.2 - для визуализации
- seaborn==0.12.1 - для создания красивых графиков
- xgboost==1.7.6 - для реализации алгоритма градиентного бустинга
- ucimlrepo==0.0.3 - для загрузки датасета из UCI ML Repository
- plotly==5.14.1 - для интерактивных графиков
- joblib==1.2.0 - для сохранения и загрузки моделей
- streamlit-option-menu==0.3.2 - для создания меню навигации

## Контакты

Автор проекта: [Александр]  
Email: [khabir-ex@mail.ru ]  
GitHub: [https://github.com/ilinAR4/predictive_maintenance_project]
