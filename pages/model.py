import matplotlib.pyplot as plt
import streamlit as st
import train_model_func
import pandas as pd
import numpy as np
import time

plt.style.use('dark_background')
my_colors = ['#2350D9', '#417CF2', '#25D997', '#96D9B3', '#F2C84B']

st.set_page_config(
    page_title="Model testing",
    page_icon=":memo:",

)
data = pd.read_csv('data/final.csv')
st.title("Propensity model for bank")
st.subheader(':blue[Training model]')

st.markdown(":paperclip: We trained CatBoostClassifier to predict"
            " whether the client will respond to the bank's offer of a new service")

st.markdown("Since the data is imbalanced, we optimize class threshold for best recall and precision metrics")

X_test = train_model_func.load_from_pkl(file_name='X_test')
y_test = train_model_func.load_from_pkl(file_name='y_test')
table, plot = train_model_func.visualisation_metrics(X_test, y_test, 0.15)
st.write(table)
st.pyplot(plot)

st.markdown('## :blue[Define the threshold and evaluate metrics]')
i = st.slider('Choose class threshold', 0.0, 1.0, 0.01)
table, plot = train_model_func.visualisation_metrics(X_test, y_test, i)
st.write(table)
st.pyplot(plot)

st.markdown('## :blue[Predict client propensity to bank promotions]')
st.markdown('Fill information about client')
col1, col2, col3 = st.columns(3)
df1 = pd.DataFrame({'AGREEMENT_RK': np.random.randint(59910150, 75292242, 1),
                    'AGE': [col1.number_input('Age', min_value=18, max_value=100, step=1)],
                    'GENDER': [col1.radio('Gender', [0, 1], horizontal=True)],
                    'EDUCATION': [
                        col1.selectbox('Education', ['Среднее', 'Среднее специальное', 'Высшее', 'Неполное среднее'])],
                    'MARITAL_STATUS': [
                        col1.selectbox('Marital', ['Не состоял в браке', 'Состою в браке', 'Вдовец/Вдова',
                                                   'Гражданский брак', 'Разведен(а)'])],
                    'CHILD_TOTAL': [col1.selectbox('Children', ['0', '1', '2 и больше'])],
                    'DEPENDANTS': [col2.selectbox('Dependants', ['0', '1', '2 и больше'])],
                    'SOCSTATUS_WORK_FL': [col2.radio('Work', [0, 1], horizontal=True)],
                    'SOCSTATUS_PENS_FL': [col2.radio('Pension', [0, 1], horizontal=True)],
                    'OWN_AUTO': [col2.radio('Car', [0, 1], horizontal=True)],
                    'FL_PRESENCE_FL': [col2.radio('Flat', [0, 1], horizontal=True)],
                    'FAMILY_INCOME': [
                        col3.selectbox('Family income', ['от 20000 до 50000 руб.', 'от 10000 до 20000 руб.',
                                                         'от 5000 до 10000 руб.', 'свыше 50000 руб.'])],
                    'PERSONAL_INCOME': [col3.number_input('Personal income', min_value=1000, max_value=300000)],
                    'CREDIT': [col3.number_input('Credit', min_value=1000, max_value=300000)],
                    'LOAN_NUM_TOTAL': [col3.radio('Number of loans', [1, 2, 3, 4], horizontal=True)],
                    'LOAN_NUM_CLOSED': [col3.radio('Number of closed loans', [0, 1, 2, 3, 4], horizontal=True)]
                    })

if st.button('Predict!'):
    with (st.spinner('In progress..')):
        time.sleep(1)
        X = train_model_func.preprocess_data(df1)
        pred = train_model_func.predict_on_input(X)
        st.write('Probability of client to accept bank offer')
        st.write(pred)


st.markdown('## :blue[Predict client propensity to bank promotions]')
st.markdown('Fill client ID')
id = st.number_input('Agreement_RK', 59910150, 75292242)

if id not in train_model_func.load_from_pkl(path='data', file_name='ids'):
    st.error('Something went wrong...')
else:
    features_data = data[data['AGREEMENT_RK'] == id].drop(['TARGET'], axis=1)
    X = train_model_func.preprocess_data(features_data)
    pred = train_model_func.predict_on_input(X)
    st.write('Probability of client to accept bank offer')
    st.write(pred)
