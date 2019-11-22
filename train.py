import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import seaborn as sns
import sys


def printing_results(y_test, y_pred, y_sk, house_list):
    from sklearn.metrics import accuracy_score as sk_a_s
    print(f'\nPrediction sklearn = \n{y_sk}\n')
    for i in range(4):
        print(f'Sklearn Accuracy {house_list[i]}:'
              f'{sk_a_s(np.where(y_test != i + 1, 0, y_test), np.where(y_sk != i + 1, 0, y_sk)) * 100:.2f}%')

    print(f'\n\x1b[1;30;43mSklearn Global accuracy: {sk_a_s(y_test, y_sk) * 100:.2f}% \x1b[0m\n')
    print(f'\n\x1b[1;30;42mDSLR Global accuracy: {sk_a_s(y_test, y_pred) * 100:.2f}% \x1b[0m\n')


def compare_with_sk(x_train, x_test, y_train, y_test, y_pred, house_list, num_labels, num_features):
    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=15000)
    logreg.fit(x_train[:, 1:], y_train)
    y_sk = logreg.predict(x_test[:, 1:])
    printing_results(y_test, y_pred, y_sk, house_list)

    # Save metrics
    os.makedirs('thetas', exist_ok=True)
    dict_thetas = dict()
    for i in range(num_labels):
        list_thetas = [logreg.intercept_[i], ]
        for c in range(num_features):
            list_thetas.append(float(logreg.coef_[i][c]))

        dict_thetas[f'label_{i}'] = list_thetas

    df_coefs = pd.DataFrame([dict_thetas[key] for key in dict_thetas.keys()],
                            index=[f'label_{n}' for n in range(num_labels)],
                            columns=[f'theta_{n}' for n in range(num_features + 1)])

    df_coefs.to_excel(os.path.join('thetas', f'sk_coefs.xlsx'))


def acc_score(true, pred):
    c = 0
    length = len(true)
    for i in range(length):
        if true[i] != pred[i]:
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


def save_metrics(num_labels, classifiers, list_x, num_features):
    os.makedirs('thetas', exist_ok=True)
    dict_thetas = dict()
    for f in range(num_labels):
        list_thetas = []
        for e, c in enumerate(np.nditer(classifiers[f])):
            list_thetas.append(float(c))

        dict_thetas[f'label_{f}'] = list_thetas

    df_coefs = pd.DataFrame([dict_thetas[key] for key in dict_thetas.keys()],
                            index=[f'label_{n}' for n in range(num_labels)],
                            columns=[list_x[n - 1] if n != 0 else 'theta0' for n in range(num_features + 1)])

    df_coefs.to_excel(os.path.join('thetas', f'coefs.xlsx'))


def sigmoid(z):
    return 1 / (1 + math.e ** (-z))


def cost(theta, X, y):
    predictions = sigmoid(X @ theta)
    predictions[predictions == 1] = 0.999999  # log(1)=0 causes error in division
    predictions[predictions == 0] = 0.000001  # log(0)=1 causes error in prediction
    error = -y * np.log(predictions) - (1 - y) * np.log(1 - predictions)
    return sum(error) / len(y)


def cost_gradient(theta, X, y):
    """
    Makes an hypothesis of y with the actual thetas
    :param theta: List of thetas
    :param X: The features used to make a correlation
    :param y: The y used to confirm or infirm the correlation
    :return: A list of actualised thetas
    """

    # Gets the predictied y with the actual thetas
    predictions = sigmoid(X @ theta)
    return X.transpose() @ (predictions - y) / len(y)


def gradient_descent(X, y, lr, num_features):
    """
    Makes a gradient descent for X to y
    :param X: The features used to make a correlation
    :param y: The y used to confirm or infirm the correlation
    :param lr: The learning rate (pre-defined with others tests)
    :param num_features: Number of features
    :return: A list of thetas
    """
    theta = np.zeros(num_features + 1)
    i = 0
    while (cost(theta, X, y) > 0.15 or i < 35000) and i < 50000:
        theta = theta - (lr * cost_gradient(theta, X, y))
        i += 1

    print(f'Number of iteration: {i}')
    return theta


def compute(x_train, y_train, num_labels, num_features, house_list, list_x):
    """
    Computes our data with a gradient descent to minimize the error and make a file filled with the calculated thetas
    :param x_train: Numpy array of features
    :param y_train: Numpy array of the target variable
    :param num_labels: Number of houses
    :param num_features: Number of features
    :param house_list: List of houses
    :param list_x: List of features
    :return: A classifier (an array containing all thetas for each house)
    """

    # Creates our classifiers to be filled with thetas
    classifiers = np.zeros(shape=(num_labels, num_features + 1))

    # One vs. all algorithm, for each house set all values to 0 except the house we want to get the thetas
    for c in range(1, num_labels + 1):
        print(f'Entering gradient descent for: {house_list[c - 1]}')

        # Set all values to 0 except c
        label = (y_train == c).astype(int)

        # Get the thetas using the gradient descent
        classifiers[(c - 1), :] = gradient_descent(x_train, label, 0.000035, num_features)

    # Save metrics into an .xlsx file
    save_metrics(num_labels, classifiers, list_x, num_features)

    return classifiers


def insert_ones(x):
    """
    A small function that is made to turn a DataFrame onto a numpy array. Insert a new col
     to x and set 1 all over the array
    :param x: a DataFrame
    :return: a numpy array
    """
    # This is the thetas array, so it must be sized: length * number of features + 1. (The + 1 stand for theta0)
    # Theta0 column is going to be multiplied by 1 in the cross product so it wont affect our hypothesis
    X = np.ones(shape=(x.shape[0], x.shape[1] + 1))
    X[:, 1:] = x
    return X


def formatting_train_test(x, y):
    """
    Takes two numpy array to split into samples (train and test) and resizes them
    :param x: Features
    :param y: The target variable
    :return: The samples (x_train, x_test, y_train, y_test)
    """

    # Splits samples train / test
    if args.compare or not args.evaluate:
        if args.compare:
            print(f'\nTraining and comparing model on 100% of the datasert\n')
        x_train, x_test = x, x
        y_train, y_test = y, y
    else:
        print(f'\nTraining model on 80% of the datasert')
        print(f'Evaluating model on 20% of the dataset\n')
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Just resize arrays to a one dimensional array
    y_train = np.resize(y_train, y_train.shape[0])
    y_test = np.resize(y_test, y_test.shape[0])

    # Takes a array add a column at start and fills it with 1
    x_train = insert_ones(x_train)
    x_test = insert_ones(x_test)

    return x_train, x_test, y_train, y_test


def formatting_data(df, house_list, list_col):
    """
    Takes a DataFrame and two lists of str, to shape data and defines training and testing samples
     x's and y's
    :param df: A DataFrame
    :param house_list: List of str containing elements to be classified
    :param list_col: list of columns, first the y column others = x
    :return: numpy arrays of training and testing part, and sizes of dataset
    """

    df = df[list_col[:]]
    df = df.dropna().reset_index(drop=True)

    x = df[list_col[1:]]
    x = x.to_numpy()
    y = pd.DataFrame(df['Hogwarts House'])
    for i in range(len(house_list)):
        y['Hogwarts House'] = np.where(y['Hogwarts House'] == house_list[i], i + 1, y['Hogwarts House'])
    y = y.astype('int')
    y = y.to_numpy()

    # Makes train and test samples
    x_train, x_test, y_train, y_test = formatting_train_test(x, y)

    num_features = x.shape[1]
    num_labels = 4

    return x_train, x_test, y_train, y_test, num_features, num_labels


def train(df):
    """
    Global function that take a DataFrame and train a logistic regression out of it
    :param df: A DataFrame
    """
    house_list = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    list_col = ['Hogwarts House', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Ancient Runes']

    # Defines samples to train and test if args.compare or args.evaluate
    x_train, x_test, y_train, y_test, num_features, num_labels = formatting_data(df, house_list, list_col)

    # Defines a classifier containing our thetas for each element to classify, i.e. each house
    classifiers = compute(x_train, y_train, num_labels, num_features, house_list, list_col[1:])

    # Defines an array of probabilities, and creates for each value an array to get the four probabilities
    probabilities = sigmoid(x_test @ classifiers.transpose())

    # Get the maximum value of each array in probabilities. The + 1 stands to get the right house (ordered
    # from 0 to 3 but but the prediction expect values from 1 to 4)
    y_pred = probabilities.argmax(axis=1) + 1

    if args.evaluate or args.compare:
        print(f'\ny_test =\n{y_test}\n')
        print(f'\ny_pred =\n{y_pred}\n')

        # Plots a heat map to represent the repartition of right/wrong values for each house
        plot_heat_map(y_test, y_pred)

        for i in range(4):
            print(f'DSLR Accuracy of {house_list[i]}:'
                  f'{acc_score(np.where(y_test != i + 1, 0, y_test), np.where(y_pred != i + 1, 0, y_pred)) * 100:.2f}%')

        # Makes a comparison of our model vs. Scikit-learn
        if args.compare:
            compare_with_sk(x_train, x_test, y_train, y_test, y_pred, house_list, num_labels, num_features)


def parsing():
    """
    Parses and defines parameters
    :return: _args
    """

    parser = argparse.ArgumentParser(prog='py train.py')
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
