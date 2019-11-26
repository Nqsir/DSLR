import sys
import os
import pandas as pd
import numpy as np
import argparse
from train import sigmoid, acc_score


def compare_with_sk(df, list_col, prediction):
    sk_file = os.path.join(os.getcwd(), os.path.join('thetas', 'sk_coefs.xlsx'))
    if os.path.exists(sk_file) and os.path.isfile(sk_file) and sk_file.endswith('.xlsx'):
        try:
            sk_thetas = get_thetas(sk_file)
            sk_prediction, _ = predict(df, list_col, sk_thetas)
            print(f'\n\x1b[1;30;43mSklearn prediction :\x1b[0m\n{sk_prediction}\n')
            print(f'\n\x1b[1;30;42mComparative global accuracy: '
                  f'{acc_score(prediction, sk_prediction) * 100:.2f}% \x1b[0m\n')
        except ValueError:
            sys.exit(f'\n\x1b[1;37;41m Wrong values \x1b[0m\n')
    else:
        sys.exit(print(f'\x1b[1;37;41mNo sk_coefs.xlsx found \x1b[0m\n'))


def make_prediction(csv_file, list_col, thetas, house_list):
    df = formatting_data(csv_file, list_col)
    prediction, x_test = predict(df, list_col, thetas)
    df_pred = pd.DataFrame(prediction, columns=['Hogwarts House'])
    dict_ = {}
    for i in range(len(house_list)):
        dict_[(i + 1)] = house_list[i]

    df_pred = df_pred.replace(dict_).reset_index().rename(columns={'index': 'Index'})
    df_pred.to_csv('houses.csv', index=False)

    print(f'\x1b[1;30;42mDSLR prediction :\x1b[0m\n{prediction}\n')
    return prediction, df


def predict(df, col, thetas_):
    x_ = df[col[1:]]
    x_ = x_.to_numpy()

    X = np.ones(shape=(x_.shape[0], x_.shape[1] + 1))
    X[:, 1:] = x_

    probabilities = sigmoid(X @ thetas_.transpose())

    return probabilities.argmax(axis=1) + 1, X


def formatting_data(csv_file, list_col):
    df = pd.read_csv(csv_file)
    df = df[list_col]
    df['Hogwarts House'] = np.where(df['Hogwarts House'], 0, df['Hogwarts House'])
    df = df.dropna().reset_index(drop=True)

    return df


def get_thetas(file_):
    df = pd.read_excel(file_)
    index = list(df.index)
    thetas_ = []

    for ind in range(0, len(index)):
        thetas_.append(np.array(np.array(df.iloc[ind, 1:])))

    return np.array(thetas_)


def parsing():
    """
    Parses and defines parameters
    :return: _args
    """

    parser = argparse.ArgumentParser(prog='py predict.py')
    parser.add_argument('csv_file', help='A csv file containing data')
    parser.add_argument('-c', '--compare', action='store_true', help='Comparative mode', default=False)
    _args = parser.parse_args()

    return _args


if __name__ == '__main__':
    args = parsing()

    csv_file = os.path.join(os.getcwd(), args.csv_file)
    coefs_file = os.path.join(os.getcwd(), os.path.join('thetas', 'coefs.xlsx'))
    house_list = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    list_col = ['Hogwarts House', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Ancient Runes']
    if os.path.exists(coefs_file) and os.path.isfile(coefs_file) and coefs_file.endswith('.xlsx'):
        try:
            thetas = get_thetas(coefs_file)
        except ValueError:
            sys.exit(f'\n\x1b[1;37;41m Wrong values \x1b[0m\n')
    else:
        sys.exit(print(f'\x1b[1;37;41mNo file containing thetas has been found (╯°□°）╯︵ ┻━┻ \x1b[0m\n'))

    if os.path.exists(csv_file) and os.path.isfile(csv_file) and csv_file.endswith('.csv'):
        prediction, df = make_prediction(csv_file, list_col, thetas, house_list)

        if args.compare:
            compare_with_sk(df, list_col, prediction)
    else:
        sys.exit(print(f'\x1b[1;37;41mThe selected file must be a csv file \x1b[0m\n'))
