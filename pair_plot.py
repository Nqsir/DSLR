import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def pair_plot(df):
    """
    Display a pair plot from a DataFrame
    :param df: A DataFrame
    """
    try:
        df = df.drop(columns=['Index']).dropna().reset_index(drop=True)
    except KeyError:
        pass
    plt.style.use('fast')
    sns.pairplot(df, hue='Hogwarts House')
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.93, top=0.95)


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
        pair_plot(df)

        if args.save:
            plt.savefig(os.path.join(os.getcwd(), 'pair_plot.png'))

        plt.show()

    else:
        sys.exit(print(f'\x1b[1;37;41mThe selected file must be a csv file \x1b[0m\n'))
