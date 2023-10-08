import pickle
import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

my_colors = ['#2350D9', '#417CF2', '#25D997', '#96D9B3', '#F2C84B']
plt.style.use('dark_background')

HERE = ''

def load_from_pkl(path='data', file_name='model'):
    """
    Load saved model as pickle file
    """
    with open(f'{HERE}/{path}/{file_name}.pickle', 'rb') as f:
        loading = pickle.load(f)
    return loading


def save_to_pkl(data, path='data', file_name='model'):
    """
    Save oblects as pickle files
    """
    with open(f'{HERE}/{path}/{file_name}.pickle', 'wb') as f:
        pickle.dump(data, f)


def predict_on_input(df: pd.DataFrame):
    """
    Load model and returns probability
    """
    model = load_from_pkl()
    pred = model.predict_proba(df)
    return pred


def preprocess_data(data):
    """
    Function for scaling and encoding data
    """
    col_transformer = load_from_pkl(path='data', file_name='col_transformer')
    X = col_transformer.transform(data)
    return X


def train_model(data: pd.DataFrame, classifier: str):
    """
    Function that take model and parameters from models dictionary and train it on data
    :param data: pd.DataFrame
    :param classifier: name of model
    """
    models = {
        'LogisticRegression': LogisticRegression(C=0.001, penalty=None, solver='lbfgs'),
        'CatBoost': CatBoostClassifier(learning_rate=0.1, loss_function='Logloss', eval_metric='Accuracy',
                                       iterations=500,
                                       max_depth=4, l2_leaf_reg=3, ),
        'SVM': SVC(probability=True, C=10, kernel='sigmoid')
    }

    X = data.drop(['id', 'target'], axis=1)
    X = preprocess_data(X)
    y = data['target']
    model = models[classifier]
    model.fit(X, y)

    save_to_pkl(model, file_name=f"{classifier}-model")


