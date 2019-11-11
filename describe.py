import os
import sys
import argparse
import pandas as pd
import math


def quartile_1(col, col_length):
    """
    First quartile (25%)
    :param col: column (DF format)
    :param col_length: column's length
    :return: First quartile
    """

    Q1 = (col_length - 1) / 4
    calc = ((Q1 - math.trunc(Q1)) * 100) / 25
    if Q1.is_integer():
        return col[Q1]
    else:
        return ((col[math.trunc(Q1)] * (4 - calc)) + col[math.ceil(Q1)] * calc) / 4


def quartile_2(col, col_length):
    """
    Second quartile or median (50%)
    :param col: column (DF format)
    :param col_length: column's length
    :return: Second quartile
    """

    Q2 = ((col_length - 1) / 4) * 2
    calc = ((Q2 - math.trunc(Q2)) * 100) / 25
    if Q2.is_integer():
        return col[Q2]
    else:
        return ((col[math.trunc(Q2)] * (4 - calc)) + col[math.ceil(Q2)] * calc) / 4


def quartile_3(col, col_length):
    """
    Third quartile (75%)
    :param col: column (DF format)
    :param col_length: column's length
    :return: Third quartile
    """

    Q3 = ((col_length - 1) / 4) * 3
    calc = ((Q3 - math.trunc(Q3)) * 100) / 25
    if Q3.is_integer():
        return col[Q3]
    else:
        return ((col[math.trunc(Q3)] * (4 - calc)) + col[math.ceil(Q3)] * calc) / 4


def std(col, col_length, mean):
    """
    Make the Standard deviation of the observations
    :param col: column (DF format)
    :param col_length: column's length
    :param mean: mean of the col values
    :return: the Standard deviation of the observations
    """

    return (sum([((mean - val) ** 2) for val in col]) / (col_length - 1)) ** (1 / 2)


def mean(col, col_length):
    """
    Make the mean of the col values
    :param col: column (DF format)
    :param col_length: column's length
    :return: the mean
    """

    return col.sum() / col_length


def count(col, col_length):
    """
    Count the number of non-NaN values
    :param col: column (DF format)
    :param col_length: column's length
    :return: number of non-NaN values
    """

    return col_length - col.isna().sum()


def describe(df):
    """
    Describes a DataFrame, reproduces the behavior of the pandas DataFrame method describe() i.e. Count, Mean, Std, Min,
    25%, 50% and 75% quartiles, and Max
    Doesn't take into account NaN values
    :param df: A DataFrame
    :return: a new DataFrame with metric for each numerical feature
    """

    res_count = {}
    res_mean = {}
    res_std = {}
    res_min = {}
    res_25 = {}
    res_50 = {}
    res_75 = {}
    res_max = {}
    for col_name in list(df):
        if df[col_name].dtypes == 'int64' or df[col_name].dtypes == 'float64':
            col = df[col_name].dropna().sort_values().reset_index(drop=True)
            col_length = len(col)
            res_count[col_name] = count(col, col_length)
            res_mean[col_name] = mean(col, col_length)
            res_std[col_name] = std(col, col_length, res_mean[col_name])
            res_min[col_name] = col.loc[0]
            res_25[col_name] = quartile_1(col, col_length)
            res_50[col_name] = quartile_2(col, col_length)
            res_75[col_name] = quartile_3(col, col_length)
            res_max[col_name] = col.loc[col_length - 1]
    return pd.DataFrame([res_count, res_mean, res_std, res_min, res_25, res_50, res_75, res_max],
                        index=['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'])


def parsing():
    """
    Parses and defines parameters
    :return: _args
    """

    parser = argparse.ArgumentParser(prog='py describe.py')
    parser.add_argument('csv_file', help='A csv file containing data')
    parser.add_argument('-c', '--compare', action='store_true', help='Comparative mode', default=False)
    _args = parser.parse_args()

    return _args


if __name__ == '__main__':
    # Parse argument
    args = parsing()

    file = os.path.join(os.getcwd(), args.csv_file)
    if os.path.exists(file)and os.path.isfile(file) and file.endswith('.csv'):
        df = pd.read_csv(file)
        df_described = describe(df)
        if args.compare:
            print(f'\x1b[1;30;43mPandas describe():\x1b[0m \n\n{df.describe()}\n\n')
        print(f'\x1b[1;30;42mDSLR describe.py:\x1b[0m \n\n{df_described}\n\n')
    else:
        sys.exit(print(f'\x1b[1;37;41mThe selected file must be a csv file \x1b[0m\n'))
