import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import time
from pandas.api.types import CategoricalDtype
import plotly.express as px



st.set_page_config(layout="wide")


def plot_readiness_score(df):
    trace = go.Scatter(
        x=df.index,
        y=df['readiness_score'],
        mode='markers',
        marker=dict(
            size=2,
            color='#636EFA'  # this is Plotly's default blue color
        )
    )
    layout = go.Layout(
        title="Readiness Score",
        xaxis=dict(title='Index'),
        yaxis=dict(title='Readiness Score')
    )
    fig = go.Figure(data=[trace], layout=layout)
    st.plotly_chart(fig)

cat_type = CategoricalDtype(categories=['Monday', 'Tuesday',
                                            'Wednesday',
                                            'Thursday', 'Friday',
                                            'Saturday', 'Sunday'],
                                ordered=True)

def create_features(df, label=None):
    """
    Creates time series features from datetime index.
    """
    df = df.copy()
    df['date'] = df.index
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekday'] = df['date'].dt.day_name()
    df['weekday'] = df['weekday'].astype(cat_type)
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week
    df['date_offset'] = (df.date.dt.month * 100 + df.date.dt.day - 320) % 1300

    df['season'] = pd.cut(df['date_offset'], [0, 300, 602, 900, 1300],
                          labels=['Spring', 'Summer', 'Fall', 'Winter']
                          )

    X = df[['dayofweek', 'quarter', 'month', 'year',
            'dayofyear', 'dayofmonth', 'weekofyear', 'weekday',
            'season']]
    if label:
        y = df[label]
        return X, y
    return X



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
                    data = pd.read_csv(uploaded_file, index_col=[0], parse_dates=[0])
                    st.session_state['user_data'] = data[['readiness_score', 'steps', 'sleep_in_hours', 'hr_max']]
                st.session_state['user_info'] = {'Пол': sex, 'Возраст': age, 'Вес': weight}
                st.session_state['info_submitted'] = True
                st.experimental_rerun()


def report_page():

    st.title("Отчет о метриках здоровья")
    if 'user_data' in st.session_state:
        df = st.session_state['user_data']
        plot_readiness_score(df)

        X, y = create_features(df, label='readiness_score')
        features_and_target = pd.concat([X, y], axis=1)
        # Prepare the data
        df = features_and_target.dropna()
        # Create the box plot
        fig = px.box(df,
                     x='weekday',
                     y='readiness_score',
                     color='season',
                     labels={
                         "weekday": "Day of Week",
                         "readiness_score": "Readiness score"
                     },
                     title='Readiness Score by Day of Week')

        # Add figure display to streamlit
        st.plotly_chart(fig)



    else:
        st.write("No data found!")



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
