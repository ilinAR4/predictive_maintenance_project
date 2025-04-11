import streamlit as st
from streamlit_option_menu import option_menu
import analysis_and_model
import presentation

# Настройка страницы
st.set_page_config(
    page_title="Предиктивное обслуживание оборудования",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стиль для заголовка
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Заголовок приложения
st.markdown('<p class="main-header">Предиктивное обслуживание оборудования</p>', unsafe_allow_html=True)
st.markdown("#### Проект по бинарной классификации для прогнозирования отказов оборудования")
st.markdown("---")

# Боковая панель с навигацией
with st.sidebar:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("logo.png", width=100)
    st.markdown("## Навигация")
    selected = option_menu(
        menu_title=None,
        options=["Анализ и Модель", "Презентация"],
        icons=["graph-up", "easel"],
        menu_icon="cast",
        default_index=0,
    )

# Отображение выбранной страницы
if selected == "Анализ и Модель":
    analysis_and_model.show()
elif selected == "Презентация":
    presentation.show()

# Footer с информацией
st.markdown("---")
st.markdown("### О проекте")
st.markdown("""
    Этот проект предназначен для демонстрации возможностей машинного обучения 
    в области предиктивного обслуживания оборудования. 
    Используется датасет AI4I 2020 Predictive Maintenance Dataset.
""")
