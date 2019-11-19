import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import seaborn as sns
import sys


def accuracy_score(list_1, list_2):
    c = 0
    length = len(list_1)
    for i in range(length):
        if list_1[i] != list_2[i]:
            c += 1
    return 1 - (c / length)


def train(df):
    house_list = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    list_x_test = ['Hogwarts House', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Ancient Runes']

    print(f'Test ------------------------------ {list_x_test[1:]}')
    df = df[list_x_test[:]]
    df = df.dropna().reset_index(drop=True)
    print(len(df))
    x = df[list_x_test[1:]]
    x = x.to_numpy()
    y = pd.DataFrame(df['Hogwarts House'])
    for i in range(4):
        y['Hogwarts House'] = np.where(y['Hogwarts House'] == house_list[i], i + 1, y['Hogwarts House'])

    y = y.astype('int')
    spl = 0.8
    num = len(x)
    sample = int(spl * num)
    x_train = x[:sample]
    x_test = x[sample:]
    y_train = y[:sample]
    y_test = y[sample:]

    num_examples = x_train.shape[0]
    num_features = x_train.shape[1]
    num_labels = 4
    y_train = np.resize(y_train, num_examples)
    y_test = np.resize(y_test, y_test.shape[0])

    X = np.ones(shape=(x_train.shape[0], x_train.shape[1] + 1))
    X[:, 1:] = x_train

    def sigmoid(z):
        return 1 / (1 + math.e ** (-z))

    def cost(theta, X, y):
        predictions = sigmoid(X @ theta)
        predictions[predictions == 1] = 0.999999
        predictions[predictions == 0] = 0.000001
        error = -y * np.log(predictions) - (1 - y) * np.log(1 - predictions)
        return sum(error) / len(y)

    def cost_gradient(theta, X, y):
        predictions = sigmoid(X @ theta)
        return X.transpose() @ (predictions - y) / len(y)

    def gradient_descent(X, y, alpha):
        num_features = X.shape[1]
        theta = np.zeros(num_features)
        i = 0
        while (cost(theta, X, y) > 0.15 or i < 35000) and i < 50000:
            theta = theta - (alpha * cost_gradient(theta, X, y))  # compute and record the cost
            i += 1

        print(i)
        return theta

    classifiers = np.zeros(shape=(num_labels, num_features + 1))
    num_iters = 20000
    for c in range(1, num_labels + 1):
        label = (y_train == c).astype(int)
        classifiers[(c - 1), :] = gradient_descent(X, label, 0.000035)
        print(f'Thetas = {classifiers[(c - 1), :]} for {house_list[c - 1]} with learning_rate = {0.0000008}')

    X_test = np.ones(shape=(x_test.shape[0], x_test.shape[1] + 1))
    X_test[:, 1:] = x_test
    probabilities = sigmoid(X_test @ classifiers.transpose())
    y_pred = probabilities.argmax(axis=1) + 1

    # from sklearn.linear_model import LogisticRegression
    # logreg = LogisticRegression(solver='lbfgs')
    # logreg.fit(x_train, y_train)
    # for i, _ in enumerate(logreg.intercept_):
    #     print(f'θ0 = {logreg.intercept_[i]} '
    #           f'{[f"θ{e + 1} = {coeff}" for e, coeff in enumerate(list(logreg.coef_[i]))]}\n\n')
    # y_pred = []
    # x_test = x_test.reset_index(drop=True)
    # for i in range(len(x_test)):
    #     list_y = []
    #     list_y.append(1 / (1 + math.exp(-(logreg.intercept_[0] + (x_test.loc[i, 'Astronomy'] * logreg.coef_[0][0])
    #                   + (x_test.loc[i, 'Herbology'] * logreg.coef_[0][1])
    #                   + (x_test.loc[i, 'Defense Against the Dark Arts'] * logreg.coef_[0][2])
    #                   + (x_test.loc[i, 'Ancient Runes'] * logreg.coef_[0][3])))))
    #     list_y.append(1 / (1 + math.exp(-(logreg.intercept_[1] + (x_test.loc[i, 'Astronomy'] * logreg.coef_[1][0])
    #                   + (x_test.loc[i, 'Herbology'] * logreg.coef_[1][1])
    #                   + (x_test.loc[i, 'Defense Against the Dark Arts'] * logreg.coef_[1][2])
    #                   + (x_test.loc[i, 'Ancient Runes'] * logreg.coef_[1][3])))))
    #     list_y.append(1 / (1 + math.exp(-(logreg.intercept_[2] + (x_test.loc[i, 'Astronomy'] * logreg.coef_[2][0])
    #                   + (x_test.loc[i, 'Herbology'] * logreg.coef_[2][1])
    #                   + (x_test.loc[i, 'Defense Against the Dark Arts'] * logreg.coef_[2][2])
    #                   + (x_test.loc[i, 'Ancient Runes'] * logreg.coef_[2][3])))))
    #     list_y.append(1 / (1 + math.exp(-(logreg.intercept_[3] + (x_test.loc[i, 'Astronomy'] * logreg.coef_[3][0])
    #                   + (x_test.loc[i, 'Herbology'] * logreg.coef_[3][1])
    #                   + (x_test.loc[i, 'Defense Against the Dark Arts'] * logreg.coef_[3][2])
    #                   + (x_test.loc[i, 'Ancient Runes'] * logreg.coef_[3][3])))))
    #     val_y = 0
    #     key_y = 0
    #     for e, v in enumerate(list_y):
    #         if v > val_y:
    #             val_y = v
    #             key_y = e + 1
    #
    #     y_pred.append(key_y)
    #
    # y_pred = np.array(y_pred)
    print(f'Prediction calculee a la main = \n{y_pred}\n')
    print(f'Reel = \n{y_test}\n')
    # print(f'Prediction sklearn = \n{logreg.predict(x_test)}\n')

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

    print(f'Accuracy {house_list[0]}:'
          f'{accuracy_score(np.where(y_test != 1, 0, y_test), np.where(y_pred != 1, 0, y_pred)) * 100:.2f}%')
    print(f'Accuracy {house_list[1]}:'
          f'{accuracy_score(np.where(y_test != 2, 0, y_test), np.where(y_pred != 2, 0, y_pred)) * 100:.2f}%')
    print(f'Accuracy {house_list[2]}:'
          f'{accuracy_score(np.where(y_test != 3, 0, y_test), np.where(y_pred != 3, 0, y_pred)) * 100:.2f}%')
    print(f'Accuracy {house_list[3]}:'
          f'{accuracy_score(np.where(y_test != 4, 0, y_test), np.where(y_pred != 4, 0, y_pred)) * 100:.2f}%')


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
