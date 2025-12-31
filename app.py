import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os

st.set_page_config(
    page_title="Предсказание цены автомобиля",
    page_icon="🚗",
    layout="wide"
)

st.title("Приложение для предсказания цены автомобиля")
st.markdown("---")

@st.cache_data
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
    return df

@st.cache_resource
def load_model():
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    else:
        st.warning("Файл модели не найден. Пожалуйста, обучите и сохраните модель сначала.")
        return None

def numeric_df(df): 
    if 'mileage' in df.columns and df['mileage'].dtype == 'object':
        df['mileage'] = df['mileage'].astype(str).str.replace(r'[^\d.]','', regex=True)
        df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
    
    if 'engine' in df.columns and df['engine'].dtype == 'object':
        df['engine'] = df['engine'].astype(str).str.replace(r'[^\d.]','', regex=True)
        df['engine'] = pd.to_numeric(df['engine'], errors='coerce')
    
    if 'max_power' in df.columns and df['max_power'].dtype == 'object':
        df['max_power'] = df['max_power'].astype(str).str.replace(r'[^\d.]','', regex=True)
        df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')
    
    if 'torque' in df.columns:
        df = df.drop('torque', axis=1)
    
    if 'name' in df.columns:
        df = df.drop('name', axis=1)
    
    if 'selling_price' in df.columns:
        df = df.drop('selling_price', axis=1)
    
    numeric_columns = ['year', 'km_driven', 'mileage', 'engine', 'max_power']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    if 'seats' in df.columns:
        df['seats'] = df['seats'].fillna(5) 
    
    
    if 'engine' in df.columns:
        df['engine'] = df['engine'].astype(int)
    if 'seats' in df.columns:
        df['seats'] = df['seats'].astype(int)
    return df

def preprocess_input(df, model_data):
    df = numeric_df(df)
    if 'ohe' in model_data:
        ohe = model_data['ohe']
        categorical_features = model_data.get('categorical_features', ['fuel', 'seller_type', 'transmission', 'owner', 'seats'])
        numerical_features = model_data.get('numerical_features', ['year', 'km_driven', 'mileage', 'engine', 'max_power'])
        
        cat_encoded = ohe.transform(df[categorical_features])
        cat_feature_names = ohe.get_feature_names_out(categorical_features)
        cat_df = pd.DataFrame(cat_encoded, columns=cat_feature_names, index=df.index)
        
        num_df = df[numerical_features]
        result_df = pd.concat([num_df.reset_index(drop=True), cat_df.reset_index(drop=True)], axis=1)
        
        return result_df
    
    return df

st.sidebar.title("Навигация")
page = st.sidebar.radio("Перейти к", ["EDA", "Предсказание", "Веса модели"])

df = load_data()
preprocessed_df = numeric_df(df)
if page == "EDA":
    st.header("📊 Разведочный анализ данных")
    

    st.subheader("Обзор датасета")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Всего записей", df.shape[0])
    with col2:
        st.metric("Всего признаков", df.shape[1])
    with col3:
        if 'selling_price' in df.columns:
            st.metric("Целевая переменная", "selling_price")
        else:
            st.metric("Всего признаков", df.shape[1])
    
    st.subheader("Пример данных")
    st.dataframe(preprocessed_df.head(10))
    
    if 'selling_price' not in df.columns:
        st.warning("⚠️ Целевая переменная 'selling_price' не найдена в датасете. Некоторые визуализации могут быть ограничены.")
    
    st.subheader("Распределение признаков")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.warning("Числовые столбцы не найдены в датасете.")
    else:
        if 'selling_price' in numeric_cols:
            st.markdown("### Распределение цен")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(df['selling_price'].dropna(), bins=50, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Цена продажи')
            ax.set_ylabel('Частота')
            ax.set_title('Распределение цен автомобилей')
            st.pyplot(fig)
            plt.close(fig)
        
        st.markdown("### Распределение других признаков")
        other_numeric = [col for col in numeric_cols if col != 'selling_price']
        if other_numeric:
            selected_feature = st.selectbox("Выберите признак", other_numeric)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(df[selected_feature].dropna(), bins=50, edgecolor='black', alpha=0.7, color='coral')
            ax.set_xlabel(selected_feature)
            ax.set_ylabel('Частота')
            ax.set_title(f'Распределение {selected_feature}')
            st.pyplot(fig)
            plt.close(fig)
    
    st.markdown("### Анализ категориальных признаков")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        selected_cat = st.selectbox("Выберите категориальный признак", categorical_cols)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        value_counts = df[selected_cat].value_counts().head(10)
        value_counts.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_xlabel(selected_cat)
        ax.set_ylabel('Количество')
        ax.set_title(f'Топ-10 значений для {selected_cat}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Категориальные столбцы не найдены в датасете.")
    
    if numeric_cols and len(numeric_cols) > 1:
        st.markdown("### Корреляционная матрица")
        fig, ax = plt.subplots(figsize=(12, 8))
        numeric_df = df[numeric_cols]
        correlation_matrix = numeric_df.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, center=0)
        ax.set_title('Матрица корреляций признаков')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    if 'selling_price' in numeric_cols and len(numeric_cols) > 1:
        st.markdown("### Цена vs Признаки")
        price_features = [col for col in numeric_cols if col != 'selling_price']
        if price_features:
            feature_vs_price = st.selectbox("Выберите признак для сравнения с ценой", 
                                             price_features,
                                             key='price_vs')
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df[feature_vs_price], df['selling_price'], alpha=0.5, color='green')
            ax.set_xlabel(feature_vs_price)
            ax.set_ylabel('Цена продажи')
            ax.set_title(f'Цена продажи vs {feature_vs_price}')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

elif page == "Предсказание":
    st.header("Предсказание цены автомобиля")
    
    model_data = load_model()
    
    if model_data is None:
        st.error("Модель не загружена. Пожалуйста, убедитесь, что файл 'model.pkl' существует в директории.")
    else:
        st.success("Модель успешно загружена!")
        
        st.subheader("Вариант 1: Загрузка CSV файла")
        uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")
        
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            st.write("Загруженные данные:")
            st.dataframe(input_df)
            
            
            if st.button("Предсказать цены", key='csv_predict'):
                try:
                    
                    processed_df = preprocess_input(input_df.copy(), model_data)
                    
                    
                    predictions = model_data['model'].predict(processed_df)
                    
                    
                    result_df = input_df.copy()
                    result_df['Предсказанная цена'] = predictions
                    
                    st.success("Предсказания завершены!")
                    st.dataframe(result_df)
                    

                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="Скачать предсказания как CSV",
                        data=csv,
                        file_name='predictions.csv',
                        mime='text/csv',
                    )
                except Exception as e:
                    st.error(f"Ошибка при выполнении предсказаний: {str(e)}")
                    st.write("Отладочная информация:")
                    st.write(f"Ожидаемые признаки: {model_data.get('feature_names', 'Недоступно')}")
                    st.write(f"Детали ошибки: {str(e)}")
        
        st.markdown("---")
        
        st.subheader("Вариант 2: Ручной ввод")
        
        col1, col2 = st.columns(2)
        
        with col1:
            year = st.number_input("Год", min_value=1990, max_value=2024, value=2015)
            km_driven = st.number_input("Пробег (км)", min_value=0, max_value=1000000, value=50000)
            mileage = st.number_input("Расход топлива (км/л)", min_value=0.0, max_value=50.0, value=18.0)
            engine = st.number_input("Объем двигателя (куб.см)", min_value=500, max_value=5000, value=1500)
            max_power = st.number_input("Макс. мощность (л.с.)", min_value=0.0, max_value=500.0, value=100.0)
        
        with col2:
            fuel = st.selectbox("Тип топлива", ["Diesel", "Petrol", "CNG", "LPG", "Electric"])
            seller_type = st.selectbox("Тип продавца", ["Individual", "Dealer", "Trustmark Dealer"])
            transmission = st.selectbox("Коробка передач", ["Manual", "Automatic"])
            owner = st.selectbox("Владелец", ["First Owner", "Second Owner", "Third Owner", 
                                           "Fourth & Above Owner", "Test Drive Car"])
            seats = st.number_input("Количество мест", min_value=2, max_value=10, value=5)
        
        if st.button("Предсказать цену", key='manual_predict'):
            input_data = pd.DataFrame({
                'year': [year],
                'km_driven': [km_driven],
                'fuel': [fuel],
                'seller_type': [seller_type],
                'transmission': [transmission],
                'owner': [owner],
                'mileage': [mileage],
                'engine': [int(engine)],
                'max_power': [max_power],
                'seats': [int(seats)]
            })
            
            try:
                
                processed_input = preprocess_input(input_data, model_data)
                prediction = model_data['model'].predict(processed_input)
                st.success(f"### Предсказанная цена: {prediction[0]:,.2f}")
                
            except Exception as e:
                st.error(f"Ошибка при предсказании: {str(e)}")
                st.write(input_data)

elif page == "Веса модели":
    st.header("Визуализация весов модели")
    
    model_data = load_model()
    
    if model_data is None:
        st.error("Модель не загружена. Пожалуйста, убедитесь, что файл 'model.pkl' существует в директории.")
    else:
        st.success("Модель успешно загружена!")
        
        try:
            model = model_data['model']
            feature_names = model_data.get('feature_names', [])
            
            if hasattr(model, 'named_steps'):
                regressor = model.named_steps.get('ridge', model.named_steps.get('lasso', 
                                                  model.named_steps.get('elasticnet', 
                                                  model.named_steps.get('linearregression', None))))
                if regressor and hasattr(regressor, 'coef_'):
                    coefficients = regressor.coef_
                else:
                    coefficients = None
            elif hasattr(model, 'coef_'):
                coefficients = model.coef_
            else:
                coefficients = None
            
            if coefficients is not None:
                if len(feature_names) == len(coefficients):
                    coef_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Coefficient': coefficients
                    })
                else:
                    coef_df = pd.DataFrame({
                        'Feature': [f'Feature_{i}' for i in range(len(coefficients))],
                        'Coefficient': coefficients
                    })
                
                coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
                coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)

                st.subheader("Топ-20 наиболее важных признаков")
                st.dataframe(coef_df.head(20))
                
                st.subheader("Важность признаков (Топ-20)")
                fig, ax = plt.subplots(figsize=(10, 12))
                
                top_n = min(20, len(coef_df))
                plot_df = coef_df.head(top_n)
                
                colors = ['red' if x < 0 else 'green' for x in plot_df['Coefficient']]
                ax.barh(range(top_n), plot_df['Coefficient'], color=colors, alpha=0.7)
                ax.set_yticks(range(top_n))
                ax.set_yticklabels(plot_df['Feature'])
                ax.set_xlabel('Значение коэффициента')
                ax.set_title('Топ-20 коэффициентов признаков')
                ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
                plt.tight_layout()
                st.pyplot(fig)
                
                
                st.subheader("Распределение всех коэффициентов")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(coefficients, bins=50, edgecolor='black', alpha=0.7, color='purple')
                ax.set_xlabel('Значение коэффициента')
                ax.set_ylabel('Частота')
                ax.set_title('Распределение коэффициентов модели')
                st.pyplot(fig)
                st.write(f"**Количество признаков:** {len(coefficients)}")
                
            else:
                st.warning("Коэффициенты модели не найдены. Эта визуализация работает с линейными моделями (Linear Regression, Ridge, Lasso, ElasticNet).")
                
        except Exception as e:
            st.error(f"Ошибка при визуализации весов модели: {str(e)}")

#! прошу обратить внимание, добавлено нейронкой
st.markdown("---")
st.markdown("### 📝 Инструкция")
st.markdown("""
- **Вкладка EDA**: Исследуйте датасет с различными визуализациями и статистикой
- **Вкладка Предсказание**: Делайте предсказания через загрузку CSV или ручной ввод
- **Вкладка Веса модели**: Визуализируйте важность различных признаков в модели
""")
