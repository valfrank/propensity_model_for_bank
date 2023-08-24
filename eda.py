import matplotlib.pyplot as plt
import seaborn as sns
from phik.report import plot_correlation_matrix

plt.style.use('dark_background')
my_colors = ['#2350D9', '#417CF2', '#25D997', '#96D9B3', '#F2C84B']

columns = ['AGE', 'GENDER', 'EDUCATION', 'MARITAL_STATUS',
           'CHILD_TOTAL', 'DEPENDANTS', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL',
           'OWN_AUTO', 'FL_PRESENCE_FL', 'FAMILY_INCOME', 'PERSONAL_INCOME',
           'CREDIT', 'LOAN_NUM_TOTAL', 'LOAN_NUM_CLOSED', 'TARGET']
names = ['Age', 'Gender', 'Education', 'Marital status',
         'Number of children', 'Number of dependants',
         'Work status', 'Pension status', 'Auto owner',
         'Flat owner', 'Family income', 'Personal income',
         'Last loan amount', 'Number of loans', 'Number of closed loans', 'Target']
category_dict = {key: value for (key, value) in zip(columns, names)}


def numeric_plot(df, column):
    """
    Function for distplot and boxplot visualization for selected numeric column
    :param df: Dataframe
    :param column: str
    :return: figure
    """
    f, ax = plt.subplots(1, 2, figsize=(10, 4))
    plt.suptitle(f'{category_dict[column]} distribution', fontweight='heavy', fontsize=18,
                 fontfamily='sans-serif', color=my_colors[1], y=1.07)
    sns.distplot(ax=ax[0], x=df[column], hist=True,
                 bins=40,
                 kde=True,
                 vertical=False,
                 color=my_colors[-1],
                 label=category_dict[column])

    ax[0].set_title('Histogram', size=14, weight='bold')
    ax[0].set_ylabel('')
    sns.boxplot(ax=ax[1], x=df[column], color=my_colors[-1])
    ax[1].set_title('Boxplot', size=14, weight='bold')
    ax[1].set_xlabel('')
    return f


def category_plot(df, column):
    """
    Function for pie chart and count plot visualization for selected category column
    :param df: Dataframe
    :param column: str
    :return: figure
    """
    f, ax = plt.subplots(1, 2, figsize=(8, 4))
    plt.suptitle(f'{category_dict[column]} distribution', fontweight='heavy', fontsize=18,
                 fontfamily='sans-serif', color=my_colors[2], y=1)
    df[column].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[0], shadow=True, colors=my_colors)
    ax[0].set_title('Pie chart', size=14, weight='bold')
    ax[0].set_ylabel('')
    sns.countplot(x=df[column], ax=ax[1], palette=my_colors)
    ax[1].set_title('Histogram', size=14, weight='bold')
    ax[1].set_xlabel('')
    f.tight_layout()
    f.autofmt_xdate()
    return f


def plot_target(df, column):
    """
    Function for count plot of target and selected category feature
    :param df: DataFrame
    :param column: str
    :return: figure
    """
    f, ax = plt.subplots(figsize=(8, 4))
    ax = sns.countplot(x=df[column], hue=df['TARGET'], data=df, palette=my_colors[3:])
    ax.set_title(f'Target vs {category_dict[column]}', size=18, weight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    f.tight_layout()
    return f


def scatter(df, column_x, column_y, gender):
    """
    Function for scatter plot using selected variables
    :param df: DataFrame
    :param column_x: str
    :param column_y: str
    :param gender: str
    :return: figure
    """
    if gender == 'male':
        df = df[df['GENDER'] == 1]
    elif gender == 'female':
        df = df[df['GENDER'] == 0]
    else:
        pass

    f, ax = plt.subplots()
    ax = sns.scatterplot(x=df[column_x],
                         y=df[column_y],
                         hue=df['TARGET'],
                         palette=my_colors[1:3])
    ax.set_xlabel(category_dict[column_x])
    ax.set_ylabel(category_dict[column_y])
    ax.set_title("Scatterplot of bank clients : {}".format(gender), size=18, weight='bold')
    f.tight_layout()
    return f


def heatmap_phik(df):
    """
    Function for phik correlation matrix visualisation
    :param df: DataFrame
    :return: figure
    """
    phik_overview = df.phik_matrix()
    plot_correlation_matrix(phik_overview.values,
                            x_labels=phik_overview.columns,
                            y_labels=phik_overview.index,
                            vmin=0, vmax=1, color_map="Blues",
                            title=r"correlation $\phi_K$",
                            fontsize_factor=0.9,
                            figsize=(20, 10))
    plt.tight_layout()
    return plt
