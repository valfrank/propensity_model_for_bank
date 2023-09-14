# Propensity Modeling for bank customer service
Project for boot camp 'Linear models' from AI Education: Propensity model for bank

Author: Valeriia Rostovtseva (stepikID 454375132)

## Intro

Propensity modeling is an approach that attempts to predict the likelihood that visitors, leads, and customers will perform certain actions.
The aim is to predict a customer response (positive or negative) to the bank's offer.
The bank will make an offer only to those from whom a positive response is expected.

## Data
This database stores information about bank customers and their personal data.

**Training dataset**
- 'AGE' - client's age, 
- 'GENDER' - gender of the client (1 — male, 0 — female), 
- 'EDUCATION' - education,
- 'MARITAL_STATUS' - martial status, 
- 'CHILD_TOTAL' - the number of children, 
- 'DEPENDANTS' - the number of dependents,
- 'SOCSTATUS_WORK_FL' - client's social status regarding work (1 — working, 0 — not working), 
- 'SOCSTATUS_PENS_FL' - client's social status in relation to pension (1 — pensioner, 0 — non-pensioner), 
- 'OWN_AUTO' - number of cars owned,
- 'FL_PRESENCE_FL' - the presence of an apartment in the property (1 - yes, 0 - no), 
- 'FAMILY_INCOME' - family income, 
- 'PERSONAL_INCOME' - personal income,
- 'CREDIT' - last loan amount, 
- 'LOAN_NUM_TOTAL' - the number of loans,
- 'LOAN_NUM_CLOSED' - the number of closed loans, 
- 'TARGET' - target variable: response to the marketing campaign (1 - the response was registered, 0 - there was no response).

## Plan
1. Data preprocess and feature engineering
    - Drop rows with duplicates and incorrectly filled values
    - Add columns with number of loans 
    - Treat missed values and outliers
    - Process and results are in [EDA.jpynb](https://github.com/valfrank/propensity_model_for_bank/blob/main/EDA.ipynb)
2. EDA 
   - Numeric and categorical values distribution
   - Visualization of dependencies of the target and features
   - Correlation matrix
   - Process and results are in [EDA.jpynb](https://github.com/valfrank/propensity_model_for_bank/blob/main/EDA.ipynb)
3. Train model
   - Scaling and encoding features
   - Train LogisticRegression and CatBoostClassifier
   - Hyperparameters tuning with GridsearchCV
   - Optimize threshold for best Recall/Precision metrics
   - Test best model
   - Process and results are in [Model_training.jpynb]()

**Check out the live [app](https://airline-satisfaction.streamlit.app/) here!**