import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from model_handler import ModelHandler
import os

def show():
    """Страница анализа данных и модели."""
    # Вкладки для разных этапов анализа
    tabs = st.tabs(["Загрузка данных", "Предобработка", "Модели", "Предсказание"])

    # Инициализация обработчика моделей
    if 'model_handler' not in st.session_state:
        st.session_state.model_handler = ModelHandler()
    
    # Вкладка загрузки данных
    with tabs[0]:
        show_data_loading()
    
    # Вкладка предобработки данных
    with tabs[1]:
        if 'data' in st.session_state:
            show_data_preprocessing()
        else:
            st.warning("Сначала загрузите данные на вкладке 'Загрузка данных'.")
    
    # Вкладка моделей
    with tabs[2]:
        if 'preprocessed_data' in st.session_state:
            show_models()
        else:
            st.warning("Сначала выполните предобработку данных на вкладке 'Предобработка'.")
    
    # Вкладка предсказания
    with tabs[3]:
        if 'trained_models' in st.session_state:
            show_prediction()
        else:
            st.warning("Сначала обучите модели на вкладке 'Модели'.")

def show_data_loading():
    """Функция для отображения загрузки данных."""
    st.header("Загрузка данных")
    
    # Метод загрузки данных
    load_method = st.radio(
        "Выберите метод загрузки данных:",
        ["Использовать предзагруженный датасет", "Загрузить свой файл"]
    )
    
    if load_method == "Использовать предзагруженный датасет":
        # Создаем директорию для данных, если она не существует
        if not os.path.exists('data'):
            os.makedirs('data')
        
        # Проверяем, существует ли файл данных
        default_data_path = 'data/predictive_maintenance.csv'
        
        # Если файла нет или пользователь хочет обновить данные
        if not os.path.exists(default_data_path) or st.button("Обновить данные"):
            try:
                with st.spinner("Загрузка данных из UCI ML Repository..."):
                    # Пробуем загрузить данные через ucimlrepo
                    try:
                        from ucimlrepo import fetch_ucirepo
                        
                        # Загрузка датасета
                        ai4i_2020 = fetch_ucirepo(id=601)
                        
                        # Данные
                        X = ai4i_2020.data.features
                        y = ai4i_2020.data.targets
                        
                        # Объединяем признаки и целевые переменные
                        data = pd.concat([X, y], axis=1)
                        
                        # Сохраняем в файл
                        data.to_csv(default_data_path, index=False)
                        
                        st.success("Данные успешно загружены из UCI ML Repository!")
                    except Exception as e:
                        st.error(f"Ошибка при загрузке данных из UCI ML Repository: {e}")
                        st.info("Используем тестовый датасет.")
                        
                        # Если не удалось загрузить из UCI, используем тестовый датасет
                        # Генерация тестовых данных для примера
                        # (этот код будет работать только при ошибке загрузки из UCI)
                        np.random.seed(42)
                        n_samples = 1000
                        
                        data = pd.DataFrame({
                            'UDI': range(1, n_samples + 1),
                            'Product ID': [f"L{i}" for i in range(n_samples)],
                            'Type': np.random.choice(['L', 'M', 'H'], size=n_samples, p=[0.5, 0.3, 0.2]),
                            'Air temperature [K]': np.random.normal(300, 2, n_samples),
                            'Process temperature [K]': np.random.normal(310, 1, n_samples),
                            'Rotational speed [rpm]': np.random.normal(1500, 100, n_samples),
                            'Torque [Nm]': np.abs(np.random.normal(40, 10, n_samples)),
                            'Tool wear [min]': np.random.randint(0, 250, n_samples)
                        })
                        
                        # Генерация целевых переменных
                        failures = np.zeros(n_samples, dtype=int)
                        twf = np.zeros(n_samples, dtype=int)
                        hdf = np.zeros(n_samples, dtype=int)
                        pwf = np.zeros(n_samples, dtype=int)
                        osf = np.zeros(n_samples, dtype=int)
                        rnf = np.zeros(n_samples, dtype=int)
                        
                        # Tool Wear Failure (TWF)
                        twf_indices = (data['Tool wear [min]'] > 200) & (data['Tool wear [min]'] < 240)
                        twf[twf_indices] = 1
                        
                        # Heat Dissipation Failure (HDF)
                        hdf_indices = ((data['Process temperature [K]'] - data['Air temperature [K]']) < 8.6) & (data['Rotational speed [rpm]'] < 1380)
                        hdf[hdf_indices] = 1
                        
                        # Power Failure (PWF)
                        power = data['Rotational speed [rpm]'] * data['Torque [Nm]'] / 9.5488
                        pwf_indices = (power < 3500) | (power > 9000)
                        pwf[pwf_indices] = 1
                        
                        # Overstrain Failure (OSF)
                        tool_torque = data['Tool wear [min]'] * data['Torque [Nm]']
                        thresholds = {'L': 11000, 'M': 12000, 'H': 13000}
                        for t, threshold in thresholds.items():
                            osf_indices = (data['Type'] == t) & (tool_torque > threshold)
                            osf[osf_indices] = 1
                        
                        # Random Failures (RNF)
                        rnf_indices = np.random.rand(n_samples) < 0.001
                        rnf[rnf_indices] = 1
                        
                        # Machine failure
                        failures = np.maximum.reduce([twf, hdf, pwf, osf, rnf])
                        
                        data['Machine failure'] = failures
                        data['TWF'] = twf
                        data['HDF'] = hdf
                        data['PWF'] = pwf
                        data['OSF'] = osf
                        data['RNF'] = rnf
                        
                        data.to_csv(default_data_path, index=False)
                        st.success("Тестовый датасет успешно создан!")
            
            except Exception as e:
                st.error(f"Ошибка при создании тестового датасета: {e}")
        
        # Загрузка данных из файла
        try:
            data = pd.read_csv(default_data_path)
            st.session_state.data = data
            st.session_state.model_handler.load_data(default_data_path)
            st.success(f"Данные успешно загружены из {default_data_path}!")
        except Exception as e:
            st.error(f"Ошибка при чтении файла {default_data_path}: {e}")
    
    else:  # Загрузка своего файла
        uploaded_file = st.file_uploader("Загрузите CSV-файл с данными", type="csv")
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                
                # Сохраняем загруженный файл в директорию data
                if not os.path.exists('data'):
                    os.makedirs('data')
                
                user_data_path = 'data/user_data.csv'
                data.to_csv(user_data_path, index=False)
                
                st.session_state.model_handler.load_data(user_data_path)
                st.success("Файл успешно загружен!")
            except Exception as e:
                st.error(f"Ошибка при чтении файла: {e}")
    
    # Отображение данных, если они загружены
    if 'data' in st.session_state:
        data = st.session_state.data
        
        st.subheader("Просмотр данных")
        
        # Выводим информацию о размере датасета
        st.write(f"Размер датасета: {data.shape[0]} строк, {data.shape[1]} столбцов")
        
        # Показываем первые несколько строк
        st.write("Первые 5 строк датасета:")
        st.dataframe(data.head())
        
        # Статистика по числовым столбцам
        st.subheader("Статистика по числовым признакам")
        st.dataframe(data.describe())
        
        # Распределение целевой переменной
        st.subheader("Распределение целевой переменной")
        target_counts = data["Machine failure"].value_counts().reset_index()
        target_counts.columns = ["Отказ", "Количество"]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x="Отказ", y="Количество", data=target_counts, ax=ax)
        ax.set_title("Распределение целевой переменной 'Machine failure'")
        ax.set_xlabel("Отказ оборудования (0 - нет, 1 - да)")
        ax.set_ylabel("Количество")
        
        for i, v in enumerate(target_counts["Количество"]):
            ax.text(i, v + 5, str(v), ha='center')
        
        st.pyplot(fig)
        
        # Показать подробную информацию о признаках
        if st.checkbox("Показать информацию о признаках"):
            st.subheader("Информация о признаках")
            
            feature_info = pd.DataFrame({
                "Признак": [
                    "UDI", 
                    "Product ID", 
                    "Type", 
                    "Air temperature [K]", 
                    "Process temperature [K]", 
                    "Rotational speed [rpm]", 
                    "Torque [Nm]", 
                    "Tool wear [min]", 
                    "Machine failure", 
                    "TWF", 
                    "HDF", 
                    "PWF", 
                    "OSF", 
                    "RNF"
                ],
                "Описание": [
                    "Уникальный идентификатор записи",
                    "Идентификатор продукта (L, M, H) и серийный номер",
                    "Тип продукта (L, M, H)",
                    "Температура окружающей среды",
                    "Рабочая температура процесса",
                    "Скорость вращения",
                    "Крутящий момент",
                    "Износ инструмента",
                    "Бинарная метка: 1 — отказ оборудования, 0 — отказ не произошел",
                    "Отказ из-за износа инструмента (Tool Wear Failure)",
                    "Отказ из-за недостаточного теплоотвода (Heat Dissipation Failure)",
                    "Отказ из-за недостаточной или избыточной мощности (Power Failure)",
                    "Отказ из-за перегрузки (Overstrain Failure)",
                    "Случайный отказ (Random Failure)"
                ],
                "Тип данных": [
                    "Integer",
                    "Categorical",
                    "Categorical",
                    "Continuous",
                    "Continuous",
                    "Integer",
                    "Continuous",
                    "Integer",
                    "Integer",
                    "Integer",
                    "Integer",
                    "Integer",
                    "Integer",
                    "Integer"
                ],
                "Единицы измерения": [
                    "-",
                    "-",
                    "-",
                    "Кельвины (K)",
                    "Кельвины (K)",
                    "обороты/мин (rpm)",
                    "Ньютон-метры (Nm)",
                    "минуты (min)",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-"
                ]
            })
            
            st.dataframe(feature_info)

def show_data_preprocessing():
    """Функция для отображения предобработки данных."""
    st.header("Предобработка данных")
    
    if 'data' not in st.session_state:
        st.warning("Сначала загрузите данные на вкладке 'Загрузка данных'.")
        return
    
    data = st.session_state.data
    
    # Опции предобработки
    st.subheader("Шаги предобработки")
    
    # Удаление ненужных столбцов
    st.write("1. Удаление ненужных столбцов")
    
    id_columns = st.multiselect(
        "Выберите ID-столбцы для удаления:",
        options=['UDI', 'Product ID'],
        default=['UDI', 'Product ID']
    )
    
    other_target_columns = st.multiselect(
        "Выберите дополнительные целевые столбцы для удаления:",
        options=['TWF', 'HDF', 'PWF', 'OSF', 'RNF'],
        default=['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    )
    
    # Кодирование категориальных переменных
    st.write("2. Кодирование категориальных переменных")
    
    categorical_columns = st.multiselect(
        "Выберите категориальные столбцы для кодирования:",
        options=['Type'],
        default=['Type']
    )
    
    # Масштабирование числовых признаков
    st.write("3. Масштабирование числовых признаков")
    
    numerical_columns = st.multiselect(
        "Выберите числовые столбцы для масштабирования:",
        options=['Air temperature [K]', 'Process temperature [K]', 
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'],
        default=['Air temperature [K]', 'Process temperature [K]', 
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    )
    
    # Кнопка для запуска предобработки
    if st.button("Выполнить предобработку"):
        with st.spinner("Выполняется предобработка данных..."):
            # Обновляем параметры в обработчике моделей
            model_handler = st.session_state.model_handler
            model_handler.id_columns = id_columns
            model_handler.categorical_columns = categorical_columns
            model_handler.numerical_columns = numerical_columns
            model_handler.other_target_columns = other_target_columns
            
            # Выполняем предобработку
            preprocessed_data = model_handler.preprocess_data()
            
            # Сохраняем предобработанные данные в session_state
            st.session_state.preprocessed_data = preprocessed_data
            
            st.success("Предобработка данных успешно выполнена!")
    
    # Отображение предобработанных данных, если они есть
    if 'preprocessed_data' in st.session_state:
        preprocessed_data = st.session_state.preprocessed_data
        
        st.subheader("Предобработанные данные")
        
        # Выводим информацию о размере датасета
        st.write(f"Размер датасета после предобработки: {preprocessed_data.shape[0]} строк, {preprocessed_data.shape[1]} столбцов")
        
        # Показываем первые несколько строк
        st.write("Первые 5 строк предобработанных данных:")
        st.dataframe(preprocessed_data.head())
        
        # Корреляционная матрица
        st.subheader("Корреляционная матрица")
        
        corr = preprocessed_data.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        plt.title("Корреляционная матрица признаков")
        st.pyplot(fig)
        
        # Анализ зависимости целевой переменной от признаков
        st.subheader("Анализ зависимости целевой переменной от признаков")
        
        selected_feature = st.selectbox(
            "Выберите признак для анализа:",
            options=[col for col in preprocessed_data.columns if col != 'Machine failure']
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if selected_feature in numerical_columns:
            # Для числовых признаков используем box plot
            sns.boxplot(x='Machine failure', y=selected_feature, 
                       data=preprocessed_data, ax=ax)
            ax.set_title(f"Зависимость {selected_feature} от наличия отказа")
            ax.set_xlabel("Отказ оборудования (0 - нет, 1 - да)")
            ax.set_ylabel(selected_feature)
        else:
            # Для категориальных признаков используем count plot
            crosstab = pd.crosstab(preprocessed_data[selected_feature], 
                                   preprocessed_data['Machine failure'])
            crosstab.plot(kind='bar', stacked=True, ax=ax)
            ax.set_title(f"Зависимость отказа от {selected_feature}")
            ax.set_xlabel(selected_feature)
            ax.set_ylabel("Количество")
            ax.legend(title="Отказ", labels=["Нет", "Да"])
        
        st.pyplot(fig)

def show_models():
    """Функция для отображения обучения и оценки моделей."""
    st.header("Обучение и оценка моделей")
    
    if 'preprocessed_data' not in st.session_state:
        st.warning("Сначала выполните предобработку данных на вкладке 'Предобработка'.")
        return
    
    preprocessed_data = st.session_state.preprocessed_data
    
    # Настройки разделения данных и обучения моделей
    st.subheader("Настройки обучения")
    
    test_size = st.slider("Доля тестовой выборки", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random state", 0, 100, 42)
    
    # Кнопка для запуска обучения моделей
    if st.button("Обучить модели"):
        with st.spinner("Выполняется обучение моделей..."):
            # Разделение данных
            model_handler = st.session_state.model_handler
            X_train, X_test, y_train, y_test = model_handler.split_data(
                preprocessed_data, test_size=test_size, random_state=random_state
            )
            
            # Обучение моделей
            models = model_handler.train_models()
            
            # Оценка моделей
            model_results = model_handler.evaluate_models()
            
            # Сохраняем результаты в session_state
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.trained_models = models
            st.session_state.model_results = model_results
            st.session_state.best_model_name = model_handler.best_model_name
            
            st.success(f"Модели успешно обучены! Лучшая модель: {model_handler.best_model_name}")
    
    # Отображение результатов обучения моделей, если они есть
    if 'trained_models' in st.session_state:
        st.subheader("Результаты обучения моделей")
        
        # Отображение метрик моделей
        comparison_df = st.session_state.model_handler.get_models_comparison_df()
        
        # Форматируем датафрейм для лучшего отображения
        formatted_df = comparison_df.copy()
        for col in ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}")
        
        st.write("Сравнение метрик моделей:")
        
        # Отмечаем лучшую модель
        def highlight_best(row):
            return ['background-color: #90EE90' if row.name == comparison_df[comparison_df['Лучшая']].index[0] else '' for _ in row]
        
        st.dataframe(formatted_df.style.apply(highlight_best, axis=1).hide(axis="columns", subset=['Лучшая']))
        # Матрица ошибок для выбранной модели
        st.subheader("Матрица ошибок")
        
        selected_model = st.selectbox(
            "Выберите модель для отображения матрицы ошибок:",
            options=list(st.session_state.trained_models.keys())
        )
        
        conf_matrix_fig = st.session_state.model_handler.plot_confusion_matrix(selected_model)
        st.pyplot(conf_matrix_fig)
        
        # ROC-кривые
        st.subheader("ROC-кривые")
        
        roc_fig = st.session_state.model_handler.plot_roc_curves()
        st.pyplot(roc_fig)
        
        # Важность признаков (если доступна)
        feature_importance_figs = st.session_state.model_handler.plot_feature_importance()
        
        if feature_importance_figs:
            st.subheader("Важность признаков")
            
            for model_name, fig in feature_importance_figs.items():
                st.write(f"Важность признаков для модели {model_name}:")
                st.pyplot(fig)
        
        # Классификационный отчет для выбранной модели
        st.subheader("Классификационный отчет")
        
        model_results = st.session_state.model_results[selected_model]
        class_report = model_results['classification_report']
        
        st.text(class_report)

def show_prediction():
    """Функция для отображения страницы предсказания."""
    st.header("Предсказание отказа оборудования")
    
    if 'trained_models' not in st.session_state:
        st.warning("Сначала обучите модели на вкладке 'Модели'.")
        return
    
    model_handler = st.session_state.model_handler
    best_model_name = st.session_state.best_model_name
    
    st.write(f"Используется лучшая модель: **{best_model_name}**")
    
    # Форма для ввода параметров
    st.subheader("Введите параметры оборудования")
    
    with st.form("prediction_form"):
        # Тип продукта
        type_options = ["L", "M", "H"]
        product_type = st.selectbox("Тип продукта (Type)", options=type_options)
        
        # Числовые параметры
        col1, col2 = st.columns(2)
        
        with col1:
            air_temp = st.number_input("Температура окружающей среды [K]", 
                                      value=300.0, min_value=290.0, max_value=310.0, step=0.1)
            process_temp = st.number_input("Рабочая температура [K]", 
                                         value=310.0, min_value=300.0, max_value=320.0, step=0.1)
            rot_speed = st.number_input("Скорость вращения [rpm]", 
                                      value=1500, min_value=1000, max_value=2000, step=10)
        
        with col2:
            torque = st.number_input("Крутящий момент [Nm]", 
                                   value=40.0, min_value=10.0, max_value=80.0, step=0.1)
            tool_wear = st.number_input("Износ инструмента [min]", 
                                      value=100, min_value=0, max_value=250, step=1)
        
        # Кнопка предсказания
        submit_button = st.form_submit_button("Предсказать")
    
    # Обработка предсказания при нажатии кнопки
    if submit_button:
        # Создаем DataFrame с введенными параметрами
        input_data = pd.DataFrame({
            'Type': [product_type],
            'Air temperature [K]': [air_temp],
            'Process temperature [K]': [process_temp],
            'Rotational speed [rpm]': [rot_speed],
            'Torque [Nm]': [torque],
            'Tool wear [min]': [tool_wear]
        })
        
        # Выполняем предсказание
        prediction, prediction_proba = model_handler.predict(input_data)
        
        # Отображаем результат
        st.subheader("Результат предсказания")
        
        if prediction[0] == 1:
            st.error("Прогноз: **Вероятен отказ оборудования**")
        else:
            st.success("Прогноз: **Отказ оборудования маловероятен**")
        
        st.write(f"Вероятность отказа: **{prediction_proba[0]:.4f}** ({prediction_proba[0]*100:.2f}%)")
        
        # Определяем рисковые факторы
        risk_factors = []
        
        # Проверка факторов риска на основе значений из описания датасета
        if tool_wear > 200:
            risk_factors.append("Высокий износ инструмента (> 200 мин)")
        
        if (process_temp - air_temp) < 8.6 and rot_speed < 1380:
            risk_factors.append("Недостаточный теплоотвод (разница температур < 8.6 K и скорость < 1380 rpm)")
        
        power = rot_speed * torque / 9.5488
        if power < 3500 or power > 9000:
            risk_factors.append(f"Проблемы с мощностью ({power:.2f} Вт вне диапазона 3500-9000 Вт)")
        
        tool_torque_product = tool_wear * torque
        thresholds = {'L': 11000, 'M': 12000, 'H': 13000}
        if tool_torque_product > thresholds[product_type]:
            risk_factors.append(f"Перегрузка (произведение износа и крутящего момента > {thresholds[product_type]} minNm)")
        
        # Отображаем факторы риска
        if risk_factors:
            st.subheader("Выявленные факторы риска:")
            for factor in risk_factors:
                st.warning(factor)
        else:
            st.info("Факторы риска не выявлены.")
