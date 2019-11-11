import argparse
import os
import sys
import pandas as pd
from describe import describe
import matplotlib.pyplot as plt


def scatter_plot(df):
    """
    Plots a scatter chart from a DataFrame based on metrics (such as Std, mean quartiles...)
    :param df: A DataFrame containing metrics
    """

    try:
        df = df.drop(columns=['Index'])
    except KeyError:
        pass

    plt.style.use('ggplot')
    col = df.columns
    colors = ['black', 'gray', 'brown', 'red', 'peru', 'yellow', 'chartreuse', 'darkgreen', 'turquoise', 'teal',
              'navy', 'magenta', 'pink']

    for e, c in enumerate(col):
        plt.scatter(df.loc[:, c], df.index, label=c, c=f'{colors[e%13]}')

    plt.legend()
    plt.tight_layout(True)


def parsing():
    """
    Parses and defines parameters
    :return: _args
    """

    parser = argparse.ArgumentParser(prog='py scatter_plot.py')
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
        scatter_plot(df_described)

        if args.save:
            plt.savefig(os.path.join(os.getcwd(), 'scatter_plot.png'))

        plt.show()
    else:
        sys.exit(print(f'\x1b[1;37;41mThe selected file must be a csv file \x1b[0m\n'))
