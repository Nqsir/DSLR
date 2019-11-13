import argparse
import os
import sys
import pandas as pd
from pandas.plotting import scatter_matrix
from describe import describe
import matplotlib.pyplot as plt
import seaborn as sns


def pair_plot(df):
    """
    :param df:
    :return:
    """
    df = df.drop(index=['Count', 'Min', 'Max'], columns=['Index'])
    print(df)
    plt.style.use('ggplot')
    sns.pairplot(df)
    # scatter_matrix(df)


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
    # Parse argument
    args = parsing()

    file = os.path.join(os.getcwd(), args.csv_file)
    if os.path.exists(file)and os.path.isfile(file) and file.endswith('.csv'):
        df = pd.read_csv(file)
        df_described = describe(df)
        pair_plot(df_described)

        if args.save:
            plt.savefig(os.path.join(os.getcwd(), 'pair_plot.png'))

        plt.show()

    else:
        sys.exit(print(f'\x1b[1;37;41mThe selected file must be a csv file \x1b[0m\n'))
