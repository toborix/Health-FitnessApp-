import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import time
from pandas.api.types import CategoricalDtype
import plotly.express as px
from prophet import Prophet
import io
from pandas.plotting import register_matplotlib_converters
from prophet.plot import plot_plotly, plot_components_plotly

register_matplotlib_converters()


def print_df_info(df):
    buf = io.StringIO()
    df.info(buf=buf)
    st.text(buf.getvalue())

def create_features(df, label=None):
    """
    Creates time series features from datetime index.
    """
    cat_type = CategoricalDtype(categories=['Monday', 'Tuesday',
                                            'Wednesday',
                                            'Thursday', 'Friday',
                                            'Saturday', 'Sunday'],
                                ordered=True)

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

def plot_target_by_day_of_week(df):

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


def create_and_train_prophet_model(df_train):
    # Prepare the data for the Prophet model
    df_train_prophet = df_train.reset_index() \
        .rename(columns={'date': 'ds', 'readiness_score': 'y'})

    # Initialize and train the model
    model = Prophet()
    model.add_regressor('steps')
    model.fit(df_train_prophet)

    return model  # return the trained model


def make_prediction_and_plot(df, df_train, df_test, model):
    df_test_prophet = df_test.reset_index().rename(columns={'date': 'ds', 'readiness_score': 'y'})
    df_test_fcst = model.predict(df_test_prophet)

    st.session_state.df_test_fcst = df_test_fcst

    df.reset_index(level=0, inplace=True)
    df = df.rename(columns={'index': 'date'})

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df['date'][:len(df_train)],
            y=df['readiness_score'][:len(df_train)],
            mode='markers',
            name='Historical'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_test_prophet['ds'],
            y=df_test_prophet['y'],
            mode='markers',
            marker=dict(color='orange'),  # Orange dots for actual values in the testing period
            name='Actual'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_test_fcst['ds'],
            y=df_test_fcst['yhat'],
            mode='lines',
            name='Predicted'
        )
    )

    # The lower bound of the confidence interval, without a legend entry
    fig.add_trace(
        go.Scatter(
            x=df_test_fcst['ds'],
            y=df_test_fcst['yhat_lower'],
            mode='lines',
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False
        )
    )

    # The upper bound of the confidence interval, which will form a filled area with the earlier trace and be labelled "Confidence Interval" in the legend
    fig.add_trace(
        go.Scatter(
            x=df_test_fcst['ds'],
            y=df_test_fcst['yhat_upper'],
            mode='lines',
            fill='tonexty',
            line=dict(width=0),
            name='Confidence Interval'
        )
    )

    fig.update_layout(
        title='Prophet Forecast with Confidence Interval',
        xaxis_title='Date',
        yaxis_title='Readiness Score'
    )

    return fig

def plot_forecast_components(model, df_test_fcst):
    components_fig = plot_components_plotly(model, df_test_fcst)
    return components_fig

def plot_forecast_components(model, df_test_fcst):
    components_fig = plot_components_plotly(model, df_test_fcst)

    # Add title to the figure
    components_fig.layout.update(title_text='Основыные, годовые и недельные тренды для показателя готовности')
    # Set autoresize=True and update the widths of the components to 100%
    components_fig.update_layout(autosize=True, width=None)

    # Increase the gap between bars (equivalent to increasing vertical spacing between different components)
    for axis in components_fig.layout:
        if 'bargap' in components_fig.layout[axis]:
            components_fig.layout[axis]['bargap'] = 1.5  # Adjust the value as needed

    return components_fig



# User interface
def user_interface(df):
    # Previous widget code remains the same

    # Button to trigger the function
    if st.button("Create and Train Prophet Model"):
        # Train/Test Split (call your previous function to split the data here)
        df_train = st.session_state['df_train']

        # Show a spinner while the model is being trained
        with st.spinner('Training the model. This may take a while...'):
            trained_model = create_and_train_prophet_model(df_train)
            time.sleep(5)  # delay for 5 seconds
        st.success('Model training finished!')

        with st.spinner('Making predictions and plotting results...'):

            df_train = st.session_state['df_train']
            df_test = st.session_state['df_test']

            fig = make_prediction_and_plot(df, df_train, df_test, trained_model)
            st.plotly_chart(fig)  # Display the plot in Streamlit

            df_test_fcst = st.session_state['df_test_fcst']
            components_fig = plot_forecast_components(trained_model, df_test_fcst)
            st.plotly_chart(components_fig)





def plot_training_test_data(df, columns_to_exclude=['sleep_in_hours', 'steps']):
    """
    Plots the two datasets 'TRAINING SET' & 'TEST SET'.

    :param df: The original DataFrame.
    :param columns_to_exclude: The columns to exclude during plotting.
    """

    # Extract unique years from the dataset assuming the DataFrame index is date-time based
    years = sorted(set(df.index.year))

    # User input for selected_year
    selected_year = st.selectbox("Select a year for the train/test split:", options=years)

    # Train/Test Split
    df_train = df.loc[df.index.year < selected_year].copy()
    df_test = df.loc[df.index.year >= selected_year].copy()

    st.session_state.df_train = df_train
    st.session_state.df_test = df_test


    # Filter out the columns to exclude
    df_test_filtered = df_test.drop(columns=columns_to_exclude)
    df_train_filtered = df_train.drop(columns=columns_to_exclude)

    # Prepare for plotting
    df_test_filtered.rename(columns={'readiness_score': 'TEST SET'}, inplace=True)
    df_train_filtered.rename(columns={'readiness_score': 'TRAINING SET'}, inplace=True)

    # Combine the datasets
    df_combined = pd.concat([df_train_filtered, df_test_filtered], axis=1)

    # Create plots for each
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_combined.index, y=df_combined['TRAINING SET'], mode='markers', name='TRAINING SET'))
    fig.add_trace(go.Scatter(x=df_combined.index, y=df_combined['TEST SET'], mode='markers', name='TEST SET'))

    fig.update_layout(title='Readiness Score', xaxis_title='Date', yaxis_title='Readiness Score')

    # Display using streamlit
    st.plotly_chart(fig)




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
                    data = pd.read_csv(uploaded_file, index_col=[0], parse_dates=[0])
                    st.session_state['user_data'] = data[['readiness_score', 'steps', 'sleep_in_hours', 'hr_max']]
                st.session_state['user_info'] = {'Пол': sex, 'Возраст': age, 'Вес': weight}
                st.session_state['info_submitted'] = True
                st.rerun()


def report_page():

    st.title("Отчет о метриках здоровья")
    if 'user_data' in st.session_state:
        df = st.session_state['user_data']


        plot_readiness_score(df)

        plot_target_by_day_of_week(df)

        plot_training_test_data(df)

        # st.dataframe(df.head())
        # print_df_info(df)

        user_interface(df)

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
