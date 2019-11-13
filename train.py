import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pandas as pd
import argparse
import os
import seaborn as sns
import sys


def train(df):
    try:
        df = df.dropna().reset_index(drop=True)
    except KeyError:
        pass

    house_dict = {'Ravenclaw': 0, 'Slytherin': 1, 'Gryffindor': 2, 'Hufflepuff': 3}
    df = df[['Hogwarts House', 'Divination', 'Transfiguration', 'Muggle Studies', 'Charms']]
    print(df)
    df['Hogwarts House'] = np.where(df['Hogwarts House'] == 'Ravenclaw', 1, df['Hogwarts House'])
    df['Hogwarts House'] = np.where(df['Hogwarts House'] == 'Slytherin', 0, df['Hogwarts House'])
    df['Hogwarts House'] = np.where(df['Hogwarts House'] == 'Gryffindor', 0, df['Hogwarts House'])
    df['Hogwarts House'] = np.where(df['Hogwarts House'] == 'Hufflepuff', 0, df['Hogwarts House'])
    print(df)
    print(df.shape)
    print(list(df.columns))
    from sklearn.linear_model import LogisticRegression
    X = df[['Divination', 'Transfiguration', 'Muggle Studies', 'Charms']]
    y = df['Hogwarts House']
    y = y.astype('int')

    # ------------- WORKS FFS !
    print(y)
    logreg = LogisticRegression(solver='lbfgs')
    logreg.fit(X, y)

    print(f'X ======\n {X}')

    df_test = pd.read_csv(os.path.join(os.getcwd(), 'dataset_test.csv'))
    df_test = df_test[['Divination', 'Transfiguration', 'Muggle Studies', 'Charms']]
    df_test = df_test.dropna().reset_index(drop=True)
    X_test = df_test
    print(X_test)

    y_pred = logreg.predict(X)
    print(y_pred)
    from sklearn import metrics
    cnf_matrix = metrics.confusion_matrix(y, y_pred)
    print(cnf_matrix)
    # END ------------- WORKS FFS !

    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    #
    # # instantiate the model (using the default parameters)
    # logreg = LogisticRegression(solver='lbfgs')
    #
    # # fit the model with data
    # logreg.fit(X_train, y_train)
    #
    # #
    # y_pred = logreg.predict(X_test)

    class_names = [0, 1]  # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

    print("Accuracy:", metrics.accuracy_score(y, y_pred))
    print("Precision:", metrics.precision_score(y, y_pred))
    print("Recall:", metrics.recall_score(y, y_pred))





    # df2 = pd.get_dummies(df, columns=['Divination', 'Transfiguration', 'Muggle Studies', 'Charms'])
    # print(list(df2.columns))
    # print(df2)
    # sns.countplot(x='Hogwarts House', data=df, palette='hls')
    # sns.countplot(y="Best Hand", data=df)
    # plt.show()

    # print(data.shape)
    # print(list(data.columns))
    # sns.countplot(x='y', data=data, palette='hls')
    # plt.show()


def parsing():
    """
    Parses and defines parameters
    :return: _args
    """

    parser = argparse.ArgumentParser(prog='py pair_plot.py')
    parser.add_argument('csv_file', help='A csv file containing data')
    parser.add_argument('-s', '--save', action='store_true', help='Saving plot', default=False)
    _args = parser.parse_args()

    return _args


if __name__ == '__main__':
    args = parsing()

    file = os.path.join(os.getcwd(), args.csv_file)
    if os.path.exists(file)and os.path.isfile(file) and file.endswith('.csv'):
        df = pd.read_csv(file)
        train(df)

        if args.save:
            plt.savefig(os.path.join(os.getcwd(), 'pair_plot.png'))

        # plt.show()

    else:
        sys.exit(print(f'\x1b[1;37;41mThe selected file must be a csv file \x1b[0m\n'))
