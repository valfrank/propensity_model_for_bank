import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import eda
import json
import requests
import streamlit as st
from streamlit_lottie import st_lottie

plt.style.use('dark_background')
my_colors = ['#2350D9', '#417CF2', '#25D997', '#96D9B3', '#F2C84B']

st.set_page_config(
        page_title="EDA of bank customers database",
        page_icon=":memo:",

    )

st.title("Propensity model for bank")
st.subheader(':blue[EDA of bank customers database]')

url = requests.get(
    "https://lottie.host/187665b8-b180-473b-b594-310a5d12a3fd/m0WdnkDsSg.json")
url_json = dict()

if url.status_code == 200:
    url_json = url.json()
else:
    print("Error in the URL")

st_lottie(url_json, speed=1.2, width=400, height=200)

df = pd.read_csv('data/final.csv')
st.markdown('We have 17 columns with different information about bank clients')
st.write(df.sample(5))

st.markdown('## :blue[Numeric Features]')
numeric = ['PERSONAL_INCOME', 'CREDIT', 'LOAN_NUM_TOTAL', 'LOAN_NUM_CLOSED', 'AGE']
st.write(df[numeric].describe())
st.markdown(':paperclip: Columns `PERSONAL_INCOME` have outliers.')

st.markdown('### Build histogram and boxplot')
selected_var = st.selectbox('Choose category',
                            numeric)

st.pyplot(eda.numeric_plot(df, selected_var))
st.markdown(':paperclip: Long-tailed distributions are observed for `PERSONAL_INCOME` and `CREDIT` columns')

st.write("---")

st.markdown('## :blue[Category Features]')
category = ['GENDER', 'EDUCATION', 'MARITAL_STATUS', 'CHILD_TOTAL', 'DEPENDANTS',
            'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL', 'OWN_AUTO',
            'FL_PRESENCE_FL', 'FAMILY_INCOME', 'TARGET']
selected_cat_var = st.selectbox('Choose category',
                                category)
st.pyplot(eda.category_plot(df, selected_cat_var))
st.markdown(':paperclip: **We have unbalanced data:** there are much more clients with no response')
st.write("---")

st.markdown('## :blue[Category features vs target]')
selected_cat_var_2 = st.selectbox('Choose feature',
                                  category[:-1])

st.pyplot(eda.plot_target(df, selected_cat_var_2))

st.write("---")

st.markdown('## :blue[Scatter plot]')
selected_x_var = st.selectbox('Choose x variable',
                              ['AGE', 'PERSONAL_INCOME', 'CREDIT', 'LOAN_NUM_TOTAL', 'LOAN_NUM_CLOSED'])
selected_y_var = st.selectbox('Choose y variable',
                              ['PERSONAL_INCOME', 'AGE', 'CREDIT', 'LOAN_NUM_TOTAL', 'LOAN_NUM_CLOSED'])
selected_gender = st.selectbox('Choose gender filter',
                               ['all', 'male', 'female'])

st.pyplot(eda.scatter(df, selected_x_var, selected_y_var, selected_gender))

st.write("---")

st.markdown('## :blue[Heatmap]')

st.pyplot(eda.heatmap_phik(df))
st.markdown(":paperclip: Age and pension/work status have strong correlation, obviously. Otherwise, there is no "
            "strong correlation between features.")
