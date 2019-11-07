import os
import sys
import argparse
import pandas as pd
import math


def describe(_file):
    df = pd.read_csv(_file)
    print(f'{df.describe()}\n\n')
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
            Q_1 = ((col_length - 1) / 4)
            Q_2 = ((col_length - 1) / 4) * 2 if (((col_length - 1) / 4) * 2).is_integer() \
                else math.trunc(((col_length - 1) / 4) * 2)
            Q_3 = ((col_length - 1) / 4) * 3 if (((col_length - 1) / 4) * 3).is_integer() \
                else math.trunc(((col_length - 1) / 4) * 3)
            print()
            print(f'raw {((col_length) / 4)}, round down {math.trunc(((col_length) / 4))}, round up {math.ceil(((col_length) / 4))}')
            res_count[col_name] = col_length - col.isna().sum()
            res_mean[col_name] = col.sum() / col_length
            res_std[col_name] = (sum([((res_mean[col_name] - val) ** 2) for val in col]) / (col_length - 1)) ** (1 / 2)
            res_min[col_name] = col.loc[0]
            if Q_1.is_integer():
                res_25[col_name] = col.loc[Q_1]
            else:
                print(f'col.loc[399] = {col.loc[399]} col.loc[400] = {col.loc[400]} /2 = {(col.loc[399]+col.loc[400])/2}')
                res_25[col_name] = col.loc[math.trunc(col_length / 4)]
            res_50[col_name] = col.loc[Q_2]
            res_75[col_name] = col.loc[Q_3]
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
    _args = parser.parse_args()

    return _args


if __name__ == '__main__':
    # Parse argument
    args = parsing()

    file = os.path.join(os.getcwd(), args.csv_file)
    if os.path.exists(file)and os.path.isfile(file) and file.endswith('.csv'):
        print(describe(file))
    else:
        sys.exit(print(f'\x1b[1;37;41mThe selected file must be a csv file \x1b[0m\n'))
