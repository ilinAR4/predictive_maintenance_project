import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, 
    roc_curve, precision_recall_curve
)


class ModelHandler:
    def __init__(self):
        """Инициализация обработчика моделей."""
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.model_results = {}
        self.best_model = None
        self.best_model_name = None
        self.target_column = 'Machine failure'
        self.numerical_columns = [
            'Air temperature [K]', 
            'Process temperature [K]', 
            'Rotational speed [rpm]', 
            'Torque [Nm]', 
            'Tool wear [min]'
        ]
        self.categorical_columns = ['Type']
        self.id_columns = ['UDI', 'Product ID']
        self.other_target_columns = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        self.le_dict = {}
        self.scaler = StandardScaler()
        
        # Создать директорию для моделей, если она не существует
        if not os.path.exists('models'):
            os.makedirs('models')
    
    def load_data(self, data_path):
        """Загрузка данных из CSV файла."""
        self.data = pd.read_csv(data_path)
        return self.data
    
    def preprocess_data(self, drop_other_targets=True):
        """Предобработка данных."""
        if self.data is None:
            raise ValueError("Данные не загружены. Сначала вызовите метод load_data().")
        
        # Копия данных для обработки
        processed_data = self.data.copy()
        
        # Кодирование категориальных признаков
        for col in self.categorical_columns:
            if col in processed_data.columns:
                le = LabelEncoder()
                processed_data[col] = le.fit_transform(processed_data[col])
                self.le_dict[col] = le
        
        # Удаление ID-колонок
        processed_data = processed_data.drop(columns=self.id_columns, errors='ignore')
        
        # Опционально удаление других целевых колонок
        if drop_other_targets:
            processed_data = processed_data.drop(columns=self.other_target_columns, errors='ignore')
        
        # Проверка на пропущенные значения
        if processed_data.isnull().sum().sum() > 0:
            print("Обнаружены пропущенные значения. Заполняем медианами...")
            for col in processed_data.columns:
                if processed_data[col].isnull().sum() > 0:
                    if col in self.numerical_columns:
                        processed_data[col] = processed_data[col].fillna(processed_data[col].median())
                    else:
                        processed_data[col] = processed_data[col].fillna(processed_data[col].mode()[0])
        
        return processed_data
    
    def split_data(self, processed_data, test_size=0.2, random_state=42):
        """Разделение данных на обучающую и тестовую выборки."""
        # Отделяем признаки и целевую переменную
        X = processed_data.drop(columns=[self.target_column])
        y = processed_data[self.target_column]
        
        # Разделение на обучающую и тестовую выборки
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Масштабирование числовых признаков
        self.X_train_scaled = self.X_train.copy()
        self.X_test_scaled = self.X_test.copy()
        
        for col in self.numerical_columns:
            if col in self.X_train.columns:
                self.X_train_scaled[col] = self.scaler.fit_transform(self.X_train[[col]])
                self.X_test_scaled[col] = self.scaler.transform(self.X_test[[col]])
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_models(self):
        """Обучение моделей."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Данные не разделены. Сначала вызовите метод split_data().")
        
        # Логистическая регрессия
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(self.X_train_scaled, self.y_train)
        self.models['Логистическая регрессия'] = lr
        
        # Случайный лес
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.X_train_scaled, self.y_train)
        self.models['Случайный лес'] = rf
        
        # XGBoost
        xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        # Заменяем проблемные символы в именах колонок для XGBoost
        X_train_xgb = self.X_train_scaled.copy()
        if isinstance(X_train_xgb, pd.DataFrame):
            X_train_xgb.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in X_train_xgb.columns]
        xgb.fit(X_train_xgb, self.y_train)
        self.models['XGBoost'] = xgb
        
        # SVM
        svm = SVC(kernel='rbf', probability=True, random_state=42)
        svm.fit(self.X_train_scaled, self.y_train)
        self.models['SVM'] = svm
        
        # Сохранение моделей
        joblib.dump(lr, 'models/logistic_regression.pkl')
        joblib.dump(rf, 'models/random_forest.pkl')
        joblib.dump(xgb, 'models/xgboost.pkl')
        joblib.dump(svm, 'models/svm.pkl')
        
        return self.models
    
    def evaluate_models(self):
        """Оценка моделей на тестовой выборке."""
        if not self.models:
            raise ValueError("Модели не обучены. Сначала вызовите метод train_models().")
        
        self.model_results = {}
        max_roc_auc = 0
        
        for name, model in self.models.items():
            # Подготовка данных для предсказания
            X_test_eval = self.X_test_scaled.copy()
            
            # Специальная обработка для XGBoost
            if name == 'XGBoost':
                X_test_eval.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in X_test_eval.columns]
            
            # Предсказания
            y_pred = model.predict(X_test_eval)
            y_prob = model.predict_proba(X_test_eval)[:, 1]
            
            # Метрики
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_prob)
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            class_report = classification_report(self.y_test, y_pred)
            
            # Сохранение результатов
            self.model_results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'confusion_matrix': conf_matrix,
                'classification_report': class_report,
                'y_pred': y_pred,
                'y_prob': y_prob
            }
            
            # Определение лучшей модели по ROC AUC
            if roc_auc > max_roc_auc:
                max_roc_auc = roc_auc
                self.best_model = model
                self.best_model_name = name
        
        return self.model_results
    
    def predict(self, input_data):
        """Предсказание на новых данных с использованием лучшей модели."""
        if self.best_model is None:
            raise ValueError("Лучшая модель не определена. Сначала вызовите метод evaluate_models().")
        
        # Предобработка входных данных
        processed_input = input_data.copy()
        
        # Кодирование категориальных признаков
        for col in self.categorical_columns:
            if col in processed_input.columns and col in self.le_dict:
                processed_input[col] = self.le_dict[col].transform(processed_input[col])
        
        # Создаем DataFrame с точно такими же колонками, как в обучающих данных
        prediction_data = pd.DataFrame(index=processed_input.index)
        
        # Копируем данные из входного DataFrame
        for col in self.X_train.columns:
            if col in processed_input.columns:
                prediction_data[col] = processed_input[col].values
        
        # Масштабирование числовых признаков
        scaled_data = prediction_data.copy()
        for col in self.numerical_columns:
            if col in scaled_data.columns:
                # Преобразуем данные в массив NumPy перед масштабированием
                # Это избегает проблемы с именами признаков
                scaled_values = self.scaler.transform(scaled_data[[col]].values.reshape(-1, 1))
                scaled_data[col] = scaled_values.flatten()
        
        # Предсказание
        if self.best_model_name == 'XGBoost':
            # Для XGBoost нужны особые имена колонок без скобок
            xgb_data = scaled_data.copy()
            xgb_data.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in xgb_data.columns]
            prediction = self.best_model.predict(xgb_data)
            prediction_proba = self.best_model.predict_proba(xgb_data)[:, 1]
        else:
            prediction = self.best_model.predict(scaled_data)
            prediction_proba = self.best_model.predict_proba(scaled_data)[:, 1]
        
        return prediction, prediction_proba
    
    def plot_confusion_matrix(self, model_name):
        """Визуализация матрицы ошибок для выбранной модели."""
        if model_name not in self.model_results:
            raise ValueError(f"Модель {model_name} не найдена в результатах.")
        
        conf_matrix = self.model_results[model_name]['confusion_matrix']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Нет отказа', 'Отказ'],
                   yticklabels=['Нет отказа', 'Отказ'])
        plt.ylabel('Истинное значение')
        plt.xlabel('Предсказанное значение')
        plt.title(f'Матрица ошибок для модели {model_name}')
        
        return fig
    
    def plot_roc_curves(self):
        """Визуализация ROC-кривых для всех моделей."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for name, results in self.model_results.items():
            y_prob = results['y_prob']
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            roc_auc = results['roc_auc']
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Случайное предсказание')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-кривые для моделей')
        plt.legend()
        
        return fig
    
    def plot_feature_importance(self):
        """Визуализация важности признаков для древовидных моделей."""
        importance_models = {
            'Случайный лес': 'Случайный лес',
            'XGBoost': 'XGBoost'
        }
        
        figs = {}
        
        for display_name, model_name in importance_models.items():
            if model_name in self.models:
                model = self.models[model_name]
                
                if hasattr(model, 'feature_importances_'):
                    feature_importance = model.feature_importances_
                    features = self.X_train.columns
                    
                    # Сортировка признаков по важности
                    indices = np.argsort(feature_importance)
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    plt.barh(range(len(indices)), feature_importance[indices])
                    plt.yticks(range(len(indices)), [features[i] for i in indices])
                    plt.xlabel('Важность признака')
                    plt.title(f'Важность признаков для модели {display_name}')
                    
                    figs[display_name] = fig
        
        return figs
    
    def get_models_comparison_df(self):
        """Получение DataFrame с сравнением метрик моделей."""
        if not self.model_results:
            raise ValueError("Результаты моделей отсутствуют. Сначала вызовите метод evaluate_models().")
        
        comparison_data = {
            'Модель': [],
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1': [],
            'ROC AUC': []
        }
        
        for name, results in self.model_results.items():
            comparison_data['Модель'].append(name)
            comparison_data['Accuracy'].append(results['accuracy'])
            comparison_data['Precision'].append(results['precision'])
            comparison_data['Recall'].append(results['recall'])
            comparison_data['F1'].append(results['f1'])
            comparison_data['ROC AUC'].append(results['roc_auc'])
        
        comparison_df = pd.DataFrame(comparison_data)
        # Выделение лучшей модели
        comparison_df['Лучшая'] = comparison_df['Модель'] == self.best_model_name
        
        return comparison_df
