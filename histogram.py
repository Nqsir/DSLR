import argparse
import os
import sys
import pandas as pd
from describe import describe
import matplotlib.pyplot as plt


def std_bar_chart_plot(df):
    """
    Plots a bar chart from a DataFrame based on Standard Derivation
    :param df: A DataFrame containing standard derivation values, index 'Std', 'Max' and 'Min' expected
    """
    try:
        df = df.drop(columns=['Index'])
    except KeyError:
        pass

    if 'Std' in df.index and 'Max' in df.index and 'Min' in df.index:
        standards = df.loc['Std', :]
        range_ = df.loc['Max', :] - df.loc['Min', :]
        perc = (standards / range_) * 100
        perc = perc.sort_values()
        plt.style.use('ggplot')
        plt.figure(figsize=[7, 6])
        plt.gca().set_ylim([0, 35])
        plt.title('Standard Deviation plot (ascending sorted)\nFrom the most to the least homogeneous')
        plt.ylabel('Percentage (Standard deviation / (max - min))')
        plt.bar(perc.index, perc, width=0.5)
        plt.xticks(rotation='vertical')
        plt.tight_layout()


def parsing():
    """
    Parses and defines parameters
    :return: _args
    """

    parser = argparse.ArgumentParser(prog='py histogram.py')
    parser.add_argument('csv_file', help='A csv file containing data')
    parser.add_argument('-s', '--save', action='store_true', help='Saving plot', default=False)
    _args = parser.parse_args()

    return _args


if __name__ == '__main__':
    args = parsing()

    file = os.path.join(os.getcwd(), args.csv_file)
    if os.path.exists(file)and os.path.isfile(file) and file.endswith('.csv'):
        df = pd.read_csv(file)
        df = df.dropna()
        if not df.empty:
            df_described = describe(df)
            std_bar_chart_plot(df_described)

            if args.save:
                plt.savefig(os.path.join(os.getcwd(), 'std_bar_chart.png'))

            plt.show()
        else:
            sys.exit(print(f'\x1b[1;37;41mEmpty DataFrame \x1b[0m\n'))
    else:
        sys.exit(print(f'\x1b[1;37;41mThe selected file must be a csv file \x1b[0m\n'))
