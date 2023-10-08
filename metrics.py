import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score

my_colors = ['#2350D9', '#417CF2', '#25D997', '#96D9B3', '#F2C84B']
plt.style.use('dark_background')

HERE = '/Users/user/Desktop/bank_recom'


def load_from_pkl(path='data', file_name='model'):
    """ Load saved model as pickle file"""
    with open(f'{HERE}/{path}/{file_name}.pickle', 'rb') as f:
        loading = pickle.load(f)
    return loading


def visualisation_metrics(classifier, X, y, i):
    """
    Create table with metrics anв confusion matrix plot depending on threshold
    """
    model = load_from_pkl(file_name=classifier)
    test_pred = model.predict_proba(X)
    classes = test_pred[:, 1] > i
    accuracy = accuracy_score(y, classes)
    recall = recall_score(y, classes)
    precision = precision_score(y, classes)
    f1 = f1_score(y, classes)
    roc_auc = roc_auc_score(y, classes)
    metrics_table = pd.DataFrame({f'Treshold = {i}': [accuracy, recall, precision, f1, roc_auc]},
                                 index=['Accuracy', 'Recall', 'Precision', 'F1-score', 'ROC-AUC'])
    matrix = confusion_matrix(y, classes)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in matrix.flatten() / np.sum(matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    fig = plt.figure(figsize=(4, 4))
    sns.heatmap(matrix, annot=labels, fmt='', cmap=my_colors, annot_kws={"size": 12}, square=True)
    plt.title('Confusion matrix', size=16, weight="bold")
    plt.show()
    return metrics_table, fig