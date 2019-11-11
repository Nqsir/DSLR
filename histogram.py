import argparse
import os
import sys
import pandas as pd
from describe import describe
import matplotlib.pyplot as plt


def plotting(df):
    try:
        df = df.drop(columns=['Index'])
    except KeyError:
        pass

    df = df.sort_values(by='Std', axis=1, ascending=True)
    plt.style.use('ggplot')
    plt.title('Standard Deviation plot (ascending sorted)\nFrom the most to the least homogeneous')
    plt.bar(list(df), df.loc['Std', :].values)
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    plt.show()


def parsing():
    """
    Parses and defines parameters
    :return: _args
    """

    parser = argparse.ArgumentParser(prog='py describe.py')
    parser.add_argument('csv_file', help='A csv file containing data')
    _args = parser.parse_args()

    return _args


if __name__ == '__main__':
    # Parse argument
    args = parsing()

    file = os.path.join(os.getcwd(), args.csv_file)
    if os.path.exists(file)and os.path.isfile(file) and file.endswith('.csv'):
        df = pd.read_csv(file)
        df_described = describe(df)
        plotting(df_described)
    else:
        sys.exit(print(f'\x1b[1;37;41mThe selected file must be a csv file \x1b[0m\n'))
