import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import seaborn as sns
import sys


def acc_score(list_1, list_2):
    c = 0
    length = len(list_1)
    for i in range(length):
        if list_1[i] != list_2[i]:
            c += 1
    return 1 - (c / length)


def plot_heat_map(y_test, y_pred):
    from sklearn import metrics
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    class_names = [0, 1]  # name  of classes
    fig, ax = plt.subplots(figsize=[7.5, 5])
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    if args.save:
        plt.savefig(os.path.join(os.getcwd(), 'prediction_matrix_plot.png'))

    plt.show()


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


def gradient_descent(X, y, alpha, num_features):
    theta = np.zeros(num_features + 1)
    i = 0
    while (cost(theta, X, y) > 0.15 or i < 3500) and i < 5000:
        theta = theta - (alpha * cost_gradient(theta, X, y))
        i += 1
    print(f'Number of iteration: {i}')
    return theta


def prediction(x_train, y_train, x_test, num_labels, num_features, house_list):
    classifiers = np.zeros(shape=(num_labels, num_features + 1))
    for c in range(1, num_labels + 1):
        print(f'Entering gradient descent for: {house_list[c - 1]}')
        label = (y_train == c).astype(int)
        classifiers[(c - 1), :] = gradient_descent(x_train, label, 0.000035, num_features)

    # Save metrics
    # CHANGED NUM FOR TESTS PUT BACK 35 000 & 50 000
    print(classifiers)
    os.makedirs('thetas', exist_ok=True)
    dict_thetas = {}
    for f in range(num_labels):
        list_thetas = []
        for e, c in enumerate(np.nditer(classifiers[f])):
            list_thetas.append(float(c))

        dict_thetas[f'label_{f}'] = list_thetas

    df_coefs = pd.DataFrame([dict_thetas[key] for key in dict_thetas.keys()],
                            index=[f'label_{n}' for n in range(num_labels)],
                            columns=[f'theta_{n}' for n in range(num_features + 1)])
    print(df_coefs)
    df_coefs.to_excel(os.path.join('thetas', f'coefs.xlsx'))

    probabilities = sigmoid(x_test @ classifiers.transpose())

    return probabilities.argmax(axis=1) + 1


def formatting_data(df, house_list, list_x_test):
    df = df[list_x_test[:]]
    df = df.dropna().reset_index(drop=True)

    x = df[list_x_test[1:]]
    x = x.to_numpy()
    y = pd.DataFrame(df['Hogwarts House'])
    for i in range(4):
        y['Hogwarts House'] = np.where(y['Hogwarts House'] == house_list[i], i + 1, y['Hogwarts House'])
    y = y.astype('int')
    y = y.to_numpy()

    if args.evaluate or args.compare:
        print(f'\nTraining model on 80% of the datasert')
        print(f'Evaluating model on 20% of the dataset\n')
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    else:
        x_train, x_test = x, x
        y_train, y_test = y, y
    y_train = np.resize(y_train, x_train.shape[0])
    y_test = np.resize(y_test, y_test.shape[0])

    num_features = x_train.shape[1]
    num_labels = 4

    X_train = np.ones(shape=(x_train.shape[0], x_train.shape[1] + 1))
    X_train[:, 1:] = x_train

    X_test = np.ones(shape=(x_test.shape[0], x_test.shape[1] + 1))
    X_test[:, 1:] = x_test

    return X_train, X_test, y_train, y_test, num_features, num_labels


def train(df):
    house_list = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    list_x_test = ['Hogwarts House', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Ancient Runes']

    x_train, x_test, y_train, y_test, num_features, num_labels = formatting_data(df, house_list, list_x_test)

    y_pred = prediction(x_train, y_train, x_test, num_labels, num_features, house_list)

    if args.evaluate or args.compare:
        print(f'y_test =\n{y_test}\n')
        print(f'y_pred =\n{y_pred}\n')
        plot_heat_map(y_test, y_pred)

        for i in range(4):
            print(f'Accuracy of {house_list[i]}:'
                  f'{acc_score(np.where(y_test != i + 1, 0, y_test), np.where(y_pred != i + 1, 0, y_pred)) * 100:.2f}%')

        if args.compare:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score as sk_a_s
            logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial')
            logreg.fit(x_train, y_train)
            y_sk = logreg.predict(x_test)
            print(f'Prediction sklearn = \n{y_sk}\n')
            for i in range(4):
                print(f'Accuracy {house_list[i]}:'
                      f'{sk_a_s(np.where(y_test != i + 1, 0, y_test), np.where(y_sk != i + 1, 0, y_sk)) * 100:.2f}%')


def parsing():
    """
    Parses and defines parameters
    :return: _args
    """

    parser = argparse.ArgumentParser(prog='py pair_plot.py')
    parser.add_argument('csv_file', help='A csv file containing data')
    parser.add_argument('-c', '--compare', action='store_true', help='Comparative mode', default=False)
    parser.add_argument('-e', '--evaluate', action='store_true', help='Evaluate the model', default=False)
    parser.add_argument('-s', '--save', action='store_true', help='Saving plot', default=False)
    _args = parser.parse_args()

    return _args


if __name__ == '__main__':
    args = parsing()

    file = os.path.join(os.getcwd(), args.csv_file)
    if os.path.exists(file)and os.path.isfile(file) and file.endswith('.csv'):
        df = pd.read_csv(file)
        train(df)

    else:
        sys.exit(print(f'\x1b[1;37;41mThe selected file must be a csv file \x1b[0m\n'))