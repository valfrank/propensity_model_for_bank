import matplotlib.pyplot as plt
import streamlit as st
import time
import requests
import pickle

BACKEND_PATH = 'http://localhost:8000'

plt.style.use('dark_background')
my_colors = ['#2350D9', '#417CF2', '#25D997', '#96D9B3', '#F2C84B']

st.set_page_config(
    page_title="Prediction",
    page_icon=":crystal_ball:",

)

with open('/Users/user/Desktop/bank_recom/data/ids.pickle', 'rb') as f:
    ids = pickle.load(f)

st.title("Propensity model for bank")
st.subheader(':blue[Testing the best model]')
st.markdown(':paperclip: Best model for test is CatBoostClassifier with ROC-AUC score 0.74')


st.markdown('## :blue[Prediction for an existing client]')
st.markdown('Fill client ID')
id = st.number_input('Agreement_RK', 59910150, 75292242)

if id not in ids:
    st.error('Something went wrong...')
else:
    client = requests.get(
        f"{BACKEND_PATH}/id?id={id}"
    )
    cl = client.json()[0]
    prediction = requests.post(
        f"{BACKEND_PATH}/predict_model",
        json=cl
    )

    st.write(prediction.json())

st.markdown('## :blue[Prediction for a new client]')
st.markdown('Fill information about client')
col1, col2, col3 = st.columns(3)
client = {
    'age': col1.number_input('Age', min_value=18, max_value=100, step=1),
    'gender': col1.radio('Gender', [0, 1], horizontal=True),
    'education':
        col1.selectbox('Education', ['Среднее', 'Среднее специальное', 'Высшее', 'Неполное среднее']),
    'marital_status':
        col1.selectbox('Marital', ['Не состоял в браке', 'Состою в браке', 'Вдовец/Вдова',
                                   'Гражданский брак', 'Разведен(а)']),
    'child_total': col1.selectbox('Children', ['0', '1', '2 и больше']),
    'dependants': col2.selectbox('Dependants', ['0', '1', '2 и больше']),
    'socstatus_work_fl': col2.radio('Work', [0, 1], horizontal=True),
    'socstatus_pens_fl': col2.radio('Pension', [0, 1], horizontal=True),
    'own_auto': col2.radio('Car', [0, 1], horizontal=True),
    'fl_presence_fl': col2.radio('Flat', [0, 1], horizontal=True),
    'family_income':
        col3.selectbox('Family income', ['от 20000 до 50000 руб.', 'от 10000 до 20000 руб.',
                                         'от 5000 до 10000 руб.', 'свыше 50000 руб.']),
    'personal_income': col3.number_input('Personal income', min_value=1000, max_value=300000),
    'credit': col3.number_input('Credit', min_value=1000, max_value=300000),
    'loan_num_total': col3.radio('Number of loans', [1, 2, 3, 4], horizontal=True),
    'loan_num_closed': col3.radio('Number of closed loans', [0, 1, 2, 3, 4], horizontal=True)
}

if st.button('Predict!'):
    with (st.spinner('In progress..')):
        time.sleep(1)
        prediction = requests.post(
            f"{BACKEND_PATH}/predict_model",
            json=client
        )
        st.write(prediction.json())
