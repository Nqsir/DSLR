import sys
import os
import pandas as pd
import numpy as np
import argparse
from train import sigmoid


def get_thetas(file_):
    df = pd.read_excel(file_)
    print(df)
    index = list(df.index)
    thetas_ = []

    for ind in range(1, len(index)):
        thetas_.append(np.array(np.array(df.iloc[ind, :])))

    return np.array(thetas_), list(df.iloc[:1, 2:])


def parsing():
    """
    Parses and defines parameters
    :return: _args
    """

    parser = argparse.ArgumentParser(prog='py predict.py')
    parser.add_argument('csv_file', help='A csv file containing data')
    _args = parser.parse_args()

    return _args


if __name__ == '__main__':
    args = parsing()

    csv_file = os.path.join(os.getcwd(), args.csv_file)
    coefs_file = os.path.join(os.getcwd(), 'thetas\coefs.xlsx')
    if os.path.exists(coefs_file) and os.path.isfile(coefs_file) and coefs_file.endswith('.xlsx'):
        try:
            thetas, list_features = get_thetas(coefs_file)
        except ValueError:
            sys.exit(f'\n\x1b[1;37;41m Wrong thetas \x1b[0m\n')
    else:
        print('\n\x1b[4;33mThetas have not been computed, set to default ¯\_(ツ)_/¯ \x1b[0m \n')
        thetas = np.array([np.array([0, 0])])
        list_features = ['null']

    print(f'Thetas = {thetas}')
    print(f'List_features = {list_features}')

    if os.path.exists(coefs_file) and os.path.isfile(coefs_file) and coefs_file.endswith('.xlsx'):
        pass
        #
        #
        #
        # probabilities = sigmoid(x_test @ classifiers.transpose())
        # if in_put < 0:
        #     sys.exit(f'\n\x1b[1;37;41m Wrong mileage \x1b[0m\n')
        # else:
        #     sys.exit(f'\n\x1b[1;30;42m The estimated price is: {theta0 + theta1 * in_put:.2f} \x1b[0m\n')
    else:
        sys.exit(print(f'\x1b[1;37;41mThe selected file must be a csv file \x1b[0m\n'))
