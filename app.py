# Standard library imports
import time

# Related third party imports
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import streamlit as st

# Local application/library specific imports
import utils

import sqlite3
from werkzeug.security import check_password_hash, generate_password_hash


class UserDb:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()

    def create_table(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, password TEXT);''')
        self.conn.commit()

    def insert_user(self, username, password):
        try:
            self.cursor.execute("INSERT INTO users VALUES (?, ?);", (username, password))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def get_user(self, username):
        self.cursor.execute("SELECT * FROM users WHERE username = ?;", (username,))
        return self.cursor.fetchone()



# database setup
user_db = UserDb('users.db')
user_db.create_table()


def register_user():
    hashed_password = generate_password_hash(st.session_state.register_password)
    if user_db.insert_user(st.session_state.register_username, hashed_password):
        st.success("Successfully registered.")
        st.session_state.user_logged = True
    else:
        st.error("This username is already registered.")


def login_user():
    data = user_db.get_user(st.session_state.login_username)
    if data is not None and check_password_hash(data[1], st.session_state.login_password):
        st.success("Successfully logged in.")
        st.session_state.user_logged = True
    else:
        st.error("The username or password you have entered is invalid.")




st.set_page_config(layout="wide")
col1, col2 = st.columns([1, 1])




register_matplotlib_converters()

def login_form():
    st.title('Login Form')
    st.text_input('Login Username', key='login_username')
    st.text_input('Login Password', type='password', key='login_password')
    st.button('Login', on_click=login_user)


def register_form():
    st.title('Registration Form')
    st.text_input('Register Username', key='register_username')
    st.text_input('Register Password', type='password', key='register_password')
    st.button('Register', on_click=register_user)


# User interface
def user_interface(df):
    # Previous widget code remains the same

    # Button to trigger the function
    if st.button("Create and Train Prophet Model", key='train_button'):
        # Train/Test Split (call your previous function to split the data here)
        df_train = st.session_state['df_train']

        # Show a spinner while the model is being trained
        with st.spinner('Training the model. This may take a while...'):
            trained_model = utils.create_and_train_prophet_model(df_train)
            st.session_state.trained_model = trained_model
            time.sleep(5)  # delay for 5 seconds
        st.success('Model training finished!')

        with st.spinner('Making predictions and plotting results...'):

            df_train = st.session_state['df_train']
            df_test = st.session_state['df_test']

            fig = utils.make_prediction_and_plot(df, df_train, df_test, trained_model)
            st.plotly_chart(fig)  # Display the plot in Streamlit



def welcome_screen():

    st.session_state.page = 'welcome_screen'
    st.write("Welcome to the application! Please choose an action:")
    if st.button('Register', key='register_button' ):
        st.session_state.view = 'register'
        st.rerun()
    elif st.button('Login', key='login_button' ):
        st.session_state.view = 'login'
        st.rerun()



def main_page():
    st.title('Ввод информации пользователя')

    requirements = r"""
    
    **Инструкция по загрузке данных**

    Для корректной работы системы, пожалуйста, загрузите CSV-файл, соответствующий следующему формату:

    - Столбцы должны содержать следующие данные:
        - `active_kcal`: Активные калории
        - `resting_kcal`: Калории в состоянии покоя
        - `steps`: Количество шагов
        - `hr_min`: Минимальное значение сердечного ритма
        - `hr_max`: Максимальное значение сердечного ритма
        - `sleep_in_hours`: Продолжительность сна в часах
    - Кроме вышеперечисленных столбцов, ваш датасет должен также содержать столбец с датой (`date`).
    
      Система автоматически вычислит дополнительные метрики на основе предоставленных данных.
    
    """

    with st.form(key="user_info_form"):
        uploaded_file = st.file_uploader("Загрузить данные о здоровье (CSV)", type="csv")

        submitted = st.form_submit_button("Отправить")
        if submitted:
            with st.spinner('Обработка данных...'):
                time.sleep(2)  # Имитация загрузки
                if uploaded_file is not None:
                    data = pd.read_csv(uploaded_file, index_col=[0], parse_dates=[0])
                    st.session_state['user_data'] = data[['readiness_score', 'steps', 'sleep_in_hours', 'hr_max']]
                st.session_state['info_submitted'] = True
                st.rerun()
    with st.expander("Требования к данным"):
        st.markdown(requirements)


if 'info_submitted' not in st.session_state:
    st.session_state['info_submitted'] = False
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'  # or any other default page



def data_page():
    st.title("Работа с данными")
    if 'user_data' in st.session_state:
        df = st.session_state['user_data']
        utils.plot_readiness_score(df)
        utils.plot_target_by_day_of_week(df)
        utils.plot_training_test_data(df)
        user_interface(df)

def report_page():
    with st.container():
        df = st.session_state['user_data']
        with col1:
            st.header("Тенденции показателя готовности `readiness_score`")
            df_test_fcst = st.session_state['df_test_fcst']
            trained_model = st.session_state['trained_model']
            components_fig = utils.plot_forecast_components(trained_model, df_test_fcst)
            st.plotly_chart(components_fig)

        with col2:
            st.header("Отчет")
            utils.generate_report(df)


def set_sidebar_button(button_name, state_value):
    if st.sidebar.button(button_name):
        st.session_state['page'] = state_value


def navigation():
    # Create navigation links
    st.sidebar.title("Навигация")
    button_to_state_value_mapping = {"Главная": "home",
                                     "Анализ данных": "data",
                                     "Отчет": "report"}

    for button_name, state_value in button_to_state_value_mapping.items():
        set_sidebar_button(button_name, state_value)

    # Add a separator before the logout button
    st.sidebar.markdown('---')

    if st.sidebar.button('Logout'):
        # Clear 'user_logged' attribute
        st.session_state.user_logged = False
        # Redirect to the welcome screen
        st.session_state.page = 'welcome_screen'
        st.write('You are now logged out.')


def display_application():
    if not hasattr(st.session_state, "user_logged"):
        st.session_state.user_logged = False
        st.session_state.page = 'welcome_screen'
    if not hasattr(st.session_state, "view"):
        st.session_state.view = None

    if st.session_state.user_logged:
        # Main part of Streamlit application (existing application)
        navigation()
        # if st.session_state['info_submitted']:
        #     data_page()
        if st.session_state['page'] == 'home':
            main_page()
        elif st.session_state['page'] == 'data':
            data_page()
        elif st.session_state['page'] == 'report':
            report_page()


    elif st.session_state.view == 'register':
        register_form()
    elif st.session_state.view == 'login':
        login_form()
    else:
        if st.session_state.page == 'welcome_screen':
            welcome_screen()


# Somewhere else in the code...
display_application()
