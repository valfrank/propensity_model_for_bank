import matplotlib.pyplot as plt
import streamlit as st
import time
from metrics import load_from_pkl, visualisation_metrics
import requests

import sys
# Insert functions path into working dir if they are not in the same working dir
sys.path.insert(1, "propensity_model_for_bank/frontend")


BACKEND_PATH = 'https://propensity-model.onrender.com'

plt.style.use('dark_background')
my_colors = ['#2350D9', '#417CF2', '#25D997', '#96D9B3', '#F2C84B']

st.set_page_config(
    page_title="Models evaluation",
    page_icon=":memo:",

)

st.title("Propensity model for bank")
st.subheader(':blue[Models Evaluation]')

st.markdown(":paperclip: Choose one of the model to train")
st.markdown("The aim is to predict whether the client will respond to the bank's offer of a new service")

classifier = st.selectbox('Classifier', ['LogisticRegression', 'CatBoost', 'SVM'])

if st.button('Train!'):
    with st.spinner('In progress..'):
        time.sleep(1)
        train = requests.get(
            f"{BACKEND_PATH}/train_model?classifier={classifier}"
        )
        st.write(train.json())

    X_test = load_from_pkl(file_name='X_test')
    y_test = load_from_pkl(file_name='y_test')
    table, plot = visualisation_metrics(f"{classifier}-model", X_test, y_test, 0.15)
    st.write(table)
    st.pyplot(plot)

    st.markdown('## :blue[Define the threshold and evaluate metrics]')
    st.markdown("Since the data is imbalanced, we can optimize class threshold for best recall and precision metrics")
    i = st.slider('Choose class threshold', 0.0, 1.0, 0.01)
    table, plot = visualisation_metrics(f"{classifier}-model", X_test, y_test, i)
    st.write(table)
    st.pyplot(plot)
