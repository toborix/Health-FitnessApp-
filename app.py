import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import time


st.set_page_config(layout="wide")


def main_page():
    st.title('Ввод информации пользователя_')
    with st.form("user_info_form"):
        sex = st.selectbox('Выберите пол', ['Мужской', 'Женский'])
        age = st.number_input('Введите возраст', min_value=12, max_value=110)
        weight = st.number_input('Введите вес (кг)', min_value=30, max_value=200)
        uploaded_file = st.file_uploader("Загрузить данные о здоровье (CSV)", type="csv")

        submitted = st.form_submit_button("Отправить")
        if submitted:
            with st.spinner('Обработка данных...'):
                time.sleep(2)  # Имитация загрузки
                if uploaded_file is not None:
                    data = pd.read_csv(uploaded_file)
                    st.session_state['user_data'] = data
                st.session_state['user_info'] = {'Пол': sex, 'Возраст': age, 'Вес': weight}
                st.session_state['info_submitted'] = True
                st.experimental_rerun()


def report_page():

    st.title("Отчет о метриках здоровья")

    # Custom CSS styles
    st.markdown("""
        <style>
            .metric-box {
                background-color: #333333;
                border-left: 5px solid #F63366;
                color: #ffffff;
                padding: 20px;
                margin: 5px 10px;
                border-radius: 10px;
                display: inline-block;
                font-size: 18px;
            }
            .metric-value {
                font-size: 28px;
                font-weight: bold;
            }
            .trend-indicator {
                padding: 10px;
                margin: 10px;
                font-size: 18px;
                text-align: center;
                color: white;
                border-radius: 10px;
            }
            .positive { background-color: #4CAF50; }
            .negative { background-color: #f44336; }
            .neutral { background-color: #FFC107; }
        </style>
    """, unsafe_allow_html=True)

    # Generate sample data
    dates = pd.date_range('2021-01-01', '2021-12-31', freq='D')
    data = pd.DataFrame({
        'Шаги': np.random.randint(1000, 10000, size=len(dates)),
        'Часы сна': np.random.normal(7, 1.5, size=len(dates)),
        'Активные ккал': np.random.randint(200, 600, size=len(dates))
    }, index=dates)

    for metric in data.columns:
        x = np.arange(len(data))
        y = data[metric].values
        slope, intercept = np.polyfit(x, y, 1)
        trend_line = slope * x + intercept

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=y, mode='lines', name='Исходные данные'))
        fig.add_trace(
            go.Scatter(x=data.index, y=trend_line, mode='lines', name='Линейный тренд', line=dict(color='red')))
        fig.update_layout(title=f"{metric.capitalize()} по времени", xaxis_title='Дата', yaxis_title=metric,
                          template='plotly_dark', autosize=True)

        st.plotly_chart(fig, use_container_width=True)

        # Layout columns
        col1, col2, col3, col_trend = st.columns([1, 1, 1, 2])

        # Metric statistics
        col1.metric("Среднее", f"{y.mean():.2f}")
        col2.metric("Максимум", f"{y.max():.0f}")
        col3.metric("Минимум", f"{y.min():.0f}")

        # Trend interpretation
        if slope > 0.1:
            message, advice, color_class = "Улучшение 📈", "Продолжайте в том же духе!", "positive"
        elif slope < -0.1:
            message, advice, color_class = "Ухудшение 📉", "Пора принять меры.", "negative"
        else:
            message, advice, color_class = "Стабильно ↗️", "Поддерживайте текущий уровень.", "neutral"

        trend_html = f"<div class='trend-indicator {color_class}'>{message}<br>{advice}</div>"
        col_trend.markdown(trend_html, unsafe_allow_html=True)


def group_health_analysis_page():
    st.title("Панель распределения метрик здоровья")  # Translated title

    # Dummy dataset with multiple users, age groups, and metrics
    np.random.seed(42)
    data = pd.DataFrame({
        'Age Group': np.random.choice(['20-30', '31-40', '41-50', '51-60', '61+'], 500),
        'Resting Heart Rate': np.random.normal(70, 10, 500),
        'BMI': np.random.normal(25, 4, 500),
        'VO2 Max': np.random.normal(40, 10, 500)
    })

    # Retrieve user info if exists and set default selection
    user_info = st.session_state.get('user_info', {})
    age = user_info.get('Возраст', None)

    # Determine preselected age group based on user's age
    if age:
        if age < 30:
            default_age_group = '20-30'
        elif age < 40:
            default_age_group = '31-40'
        elif age < 50:
            default_age_group = '41-50'
        elif age < 60:
            default_age_group = '51-60'
        else:
            default_age_group = '61+'
    else:
        default_age_group = '20-30'  # Set a default or handle appropriately.

    # User input to select the age group and metric
    age_group = st.selectbox("Выберите возрастную группу:", options=['20-30', '31-40', '41-50', '51-60', '61+'],
                             index=['20-30', '31-40', '41-50', '51-60', '61+'].index(default_age_group))
    metric = st.selectbox("Выберите метрику:", data.columns[1:])  # Translated prompt

    # User's metric value
    user_metric_value = data[(data['Age Group'] == age_group)][metric].sample(n=1).values[0]

    # Filter data for the selected age group and metric
    filtered_data = data[data['Age Group'] == age_group][metric]

    # Create histogram of the metric for the selected age group
    y, x = np.histogram(filtered_data, bins=20)
    colors = ['lightsalmon' if not x[i] <= user_metric_value < x[i + 1] else 'deepskyblue' for i in range(len(x) - 1)]
    fig = go.Figure(data=[go.Bar(x=(x[:-1] + x[1:]) / 2, y=y, marker_color=colors, width=np.diff(x))])

    # Update plot layout
    fig.update_layout(
        title=f'Распределение {metric} для возрастной группы {age_group} с выделением метрики пользователя',
        xaxis_title=metric,  # Metric name doesn't need translation.
        yaxis_title="Количество",
        template='plotly_white',
        bargap=0.05
    )
    st.plotly_chart(fig, use_container_width=True)

    # Explanation Text
    percentile = np.percentile(filtered_data, 100 * (filtered_data <= user_metric_value).mean())
    explanation = f"Ваше значение {metric} составляет {user_metric_value:.2f}, что ставит вас в {percentile:.0f} перцентиль среди вашей возрастной группы."

    if percentile > 50:
        comparison = "Это означает, что вы выше медианы для вашей возрастной группы."
    else:
        comparison = "Это означает, что вы ниже медианы для вашей возрастной группы."

    st.write(explanation)
    st.write(comparison)

def advice_page():
    st.title('Советы по здоровью')
    st.write("На основе ваших данных, вот несколько советов по здоровью:")
    if 'user_info' in st.session_state:
        user_info = st.session_state['user_info']
        if user_info['Возраст'] < 18:
            st.write("Обеспечьте сбалансированное питание, богатое фруктами и овощами.")
        elif user_info['Возраст'] < 50:
            st.write("Регулярные упражнения важны для поддержания вашего веса.")
        else:
            st.write("Рекомендуются регулярные медицинские осмотры для вашей возрастной группы.")


def navigation():
    # Create navigation links
    st.sidebar.title("Навигация")
    if st.sidebar.button("Главная"):
        st.session_state['page'] = 'home'
    if st.sidebar.button("Отчет"):
        st.session_state['page'] = 'report'
    if st.sidebar.button("Анализ здоровья групп"):  # Use your selected page name here in Russian
        st.session_state['page'] = 'group_health_analysis'
    if st.sidebar.button("Советы"):
        st.session_state['page'] = 'advice'



# Check session state for the current page
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

# Initialize session state
if 'info_submitted' not in st.session_state:
    st.session_state['info_submitted'] = False

if not st.session_state['info_submitted']:
    main_page()
else:
    navigation()  # Show navigation links
    if st.session_state['page'] == 'home':
        main_page()
    elif st.session_state['page'] == 'report':
        report_page()
    elif st.session_state['page'] == 'advice':
        advice_page()
    elif st.session_state['page'] == 'group_health_analysis':
        group_health_analysis_page()
