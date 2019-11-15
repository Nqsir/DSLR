import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import seaborn as sns
import sys


def train(df):
    try:
        df = df.drop(columns=['Index']).dropna().reset_index(drop=True)
    except KeyError:
        pass

    house_list = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    list_x_test = ['Hogwarts House', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Ancient Runes']

    print(f'Test ------------------------------ {list_x_test[1:]}')
    x = df[list_x_test[1:]]
    y = pd.DataFrame(df['Hogwarts House'])
    y['Hogwarts House'] = np.where(y['Hogwarts House'] == house_list[0], 1, y['Hogwarts House'])
    y['Hogwarts House'] = np.where(y['Hogwarts House'] == house_list[1], 2, y['Hogwarts House'])
    y['Hogwarts House'] = np.where(y['Hogwarts House'] == house_list[2], 3, y['Hogwarts House'])
    y['Hogwarts House'] = np.where(y['Hogwarts House'] == house_list[3], 4, y['Hogwarts House'])

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y['Hogwarts House'].astype('int'),
                                                        test_size=0.2, random_state=0)

    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression(solver='lbfgs')
    logreg.fit(x_train, y_train)
    print(f'θ0 = {logreg.intercept_[0]} {[f"θ{e + 1} = {coeff}" for e, coeff in enumerate(list(logreg.coef_[0]))]}\n\n')
    print(f'θ0 = {logreg.intercept_[1]} {[f"θ{e + 1} = {coeff}" for e, coeff in enumerate(list(logreg.coef_[1]))]}\n\n')
    print(f'θ0 = {logreg.intercept_[2]} {[f"θ{e + 1} = {coeff}" for e, coeff in enumerate(list(logreg.coef_[2]))]}\n\n')
    print(f'θ0 = {logreg.intercept_[3]} {[f"θ{e + 1} = {coeff}" for e, coeff in enumerate(list(logreg.coef_[3]))]}\n\n')
    y_pred = []
    x_test = x_test.reset_index(drop=True)
    for i in range(len(x_test)):
        list_y = []
        list_y.append(1 / (1 + math.exp(-(logreg.intercept_[0] + (x_test.loc[i, 'Astronomy'] * logreg.coef_[0][0])
                      + (x_test.loc[i, 'Herbology'] * logreg.coef_[0][1])
                      + (x_test.loc[i, 'Defense Against the Dark Arts'] * logreg.coef_[0][2])
                      + (x_test.loc[i, 'Ancient Runes'] * logreg.coef_[0][3])))))
        list_y.append(1 / (1 + math.exp(-(logreg.intercept_[1] + (x_test.loc[i, 'Astronomy'] * logreg.coef_[1][0])
                      + (x_test.loc[i, 'Herbology'] * logreg.coef_[1][1])
                      + (x_test.loc[i, 'Defense Against the Dark Arts'] * logreg.coef_[1][2])
                      + (x_test.loc[i, 'Ancient Runes'] * logreg.coef_[1][3])))))
        list_y.append(1 / (1 + math.exp(-(logreg.intercept_[2] + (x_test.loc[i, 'Astronomy'] * logreg.coef_[2][0])
                      + (x_test.loc[i, 'Herbology'] * logreg.coef_[2][1])
                      + (x_test.loc[i, 'Defense Against the Dark Arts'] * logreg.coef_[2][2])
                      + (x_test.loc[i, 'Ancient Runes'] * logreg.coef_[2][3])))))
        list_y.append(1 / (1 + math.exp(-(logreg.intercept_[3] + (x_test.loc[i, 'Astronomy'] * logreg.coef_[3][0])
                      + (x_test.loc[i, 'Herbology'] * logreg.coef_[3][1])
                      + (x_test.loc[i, 'Defense Against the Dark Arts'] * logreg.coef_[3][2])
                      + (x_test.loc[i, 'Ancient Runes'] * logreg.coef_[3][3])))))
        val_y = 0
        key_y = 0
        for e, v in enumerate(list_y):
            if v > val_y:
                val_y = v
                key_y = e + 1

        y_pred.append(key_y)
        # for key in range(1, 4, 1):
    y_pred = np.array(y_pred)
    print(y_pred)
    print(logreg.predict(x_test))
    # print(f'list(logreg.coef_[0]) = {list(logreg.coef_[0])}')

    from sklearn import metrics
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
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

    print(np.where(y_test != 1, 0, y_test))
    print(np.where(y_pred != 1, 0, y_pred))
    print(f'Accuracy {house_list[0]}:'
          f'{metrics.accuracy_score(np.where(y_test != 1, 0, y_test), np.where(y_pred != 1, 0, y_pred)) * 100:.2f}%')
    print(f'Accuracy {house_list[1]}:'
          f'{metrics.accuracy_score(np.where(y_test != 2, 0, y_test), np.where(y_pred != 2, 0, y_pred)) * 100:.2f}%')
    print(f'Accuracy {house_list[2]}:'
          f'{metrics.accuracy_score(np.where(y_test != 3, 0, y_test), np.where(y_pred != 3, 0, y_pred)) * 100:.2f}%')
    print(f'Accuracy {house_list[3]}:'
          f'{metrics.accuracy_score(np.where(y_test != 4, 0, y_test), np.where(y_pred != 4, 0, y_pred)) * 100:.2f}%')


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
