import pandas as pd
import streamlit as st
from pandas.api.types import CategoricalDtype


import plotly.graph_objs as go
import plotly.express as px

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly


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


def make_prediction_and_plot(_df, df_train, df_test, model):
    df_test_prophet = df_test.reset_index().rename(columns={'date': 'ds', 'readiness_score': 'y'})
    df_test_fcst = model.predict(df_test_prophet)

    st.session_state.df_test_fcst = df_test_fcst

    df = _df.copy()

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


def generate_report(df):
    """
    Generates a detailed report based on the readiness score and its components
    and displays it in a Streamlit app.

    Parameters:
    analysis_data (pd.DataFrame): DataFrame containing the readiness score analysis data.
    """

    analysis_data = df.copy()

    avg_daily_steps = analysis_data['steps'].mean()
    min_daily_steps = analysis_data['steps'].min()
    max_daily_steps = analysis_data['steps'].max()

    avg_sleep_hours = analysis_data['sleep_in_hours'].mean()
    min_sleep_hours = analysis_data['sleep_in_hours'].min()
    max_sleep_hours = analysis_data['sleep_in_hours'].max()

    avg_readiness_score = analysis_data['readiness_score'].mean()
    min_readiness_score = analysis_data['readiness_score'].min()
    max_readiness_score = analysis_data['readiness_score'].max()

    # Generate the text report
    report = f"""
    **Обзор**

    Наш прогноз фокусируется на показателе `readiness_score`, составном метрике, которая предоставляет информацию о ежедневной готовности, 
    учитывая активность, отдых и вариабельность сердечного ритма (HRV). Мы использовали модель Prophet для анализа сезонных паттернов и трендов этого 
    показателя, что позволило нам выработать практические рекомендации по улучшению ежедневной готовности.

    **Понимание показателя готовности**

    `readiness_score` рассчитывается по следующей формуле:
    - **Показатель активности**: Учитывает активные калории и шаги.
    - **Показатель отдыха**: Включает продолжительность сна и калории, сожженные в состоянии покоя.
    - **Показатель HRV**: Учитывает вариабельность между минимальным и максимальным значениями сердечного ритма.

    Оценивая эти три показателя, мы получаем общий показатель готовности:
    """

    formula = r"""
    $$
        \text{{readiness\_score}} = \frac{\text{{activity\_score}} + \text{{rest\_score}} + \text{{hrv\_score}}}{3}
    $$
    """

    report_rest = f""" 
    **Ключевые выводы**

    1. **Общий тренд**:
       - Трендовая линия показывает улучшение готовности со временем. Постоянные усилия могут постепенно улучшать общий показатель.

    2. **Ежедневные паттерны**:
       - Наблюдаются колебания готовности, с заметным снижением в праздничные периоды (например, в декабре). Средний показатель готовности составляет <span style='font-size:20px; font-weight:bold;'>{avg_readiness_score:.2f}</span>, минимальный <span style='font-size:20px; font-weight:bold;'>{min_readiness_score:.2f}</span>, максимальный <span style='font-size:20px; font-weight:bold;'>{max_readiness_score:.2f}</span>.

    3. **Недельные паттерны**:
       - Показатели выше с вторника по четверг и ниже в начале и конце недели.

    4. **Анализ компонента шагов**:
       - Увеличение физической активности коррелирует с более высокой готовностью. В среднем - <span style='font-size:20px; font-weight:bold;'>{avg_daily_steps:.0f}</span> шагов в день, минимум - <span style='font-size:20px; font-weight:bold;'>{min_daily_steps:.0f}</span> шагов, максимум - <span style='font-size:20px; font-weight:bold;'>{max_daily_steps:.0f}</span> шагов.

    **Практические рекомендации**

    1. **Последовательность - ключ к успеху**:
       - Старайтесь придерживаться режима, особенно в отношении сна и физической активности. Поддержание в среднем <span style='font-size:20px; font-weight:bold;'>{avg_daily_steps:.0f}</span> шагов и <span style='font-size:20px; font-weight:bold;'>{avg_sleep_hours:.1f}</span> часов сна за ночь способствует балансу готовности, минимум сна - <span style='font-size:20px; font-weight:bold;'>{min_sleep_hours:.1f}</span> часа.

    2. **Контроль в праздники и выходные**:
       - Будьте внимательны в праздники и выходные, планируйте легкие упражнения и поддерживайте диету для снижения падения готовности.

    3. **Середина недели**:
       - Среда и четверг - отличные дни для продуктивных занятий и тренировок.

    4. **Персональные корректировки**:
       - Отслеживайте свои показатели и корректируйте активность. Например, добавление 1,000 шагов может повысить ваш показатель.

    **Заключение**

    Анализ показывает позитивные тенденции и практические советы для оптимизации вашей готовности. Фокусируйтесь на регулярности режимов, наблюдайте за показателями в праздники и используйте пики производительности.
    """

    # Display the report
    st.markdown(report + formula + report_rest, unsafe_allow_html=True)


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
    fig.add_trace(
        go.Scatter(x=df_combined.index, y=df_combined['TRAINING SET'], mode='markers', name='TRAINING SET'))
    fig.add_trace(go.Scatter(x=df_combined.index, y=df_combined['TEST SET'], mode='markers', name='TEST SET'))

    fig.update_layout(title='Readiness Score', xaxis_title='Date', yaxis_title='Readiness Score')

    # Display using streamlit
    st.plotly_chart(fig)
