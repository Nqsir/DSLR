import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from describe import describe


def pearson_s_correlation_coefficient(df):
    """
    Calculates Pearson's correlation coefficient of each feature with others (similar to the pandas function df.corr())
    :param df: A DataFrame
    :return: A new DataFrame containing Pearson's correlation coefficient
    """

    columns = list(df)
    df_described = describe(df)
    res = {}
    for col_name in columns:
        if df[col_name].dtypes == 'int64' or df[col_name].dtypes == 'float64':
            res[col_name] = []
            for col in columns:
                if df[col].dtypes == 'int64' or df[col].dtypes == 'float64':
                    i = 0
                    j = 0
                    sum_ = 0
                    test = 0
                    while i < df_described.loc['Count', col_name] and j < df_described.loc['Count', col]:
                        sum_ += ((df.loc[i, col_name] - df_described.loc['Mean', col_name]) *
                                 (df.loc[j, col] - df_described.loc['Mean', col]))
                        test += (df.loc[j, col] - df_described.loc['Mean', col])
                        i += 1
                        j += 1
                    cov = sum_ / (i - 1)
                    res[col_name].append(cov / (df_described.loc['Std', col] * df_described.loc['Std', col_name]))

    return pd.DataFrame([v for v in list(res.values())], index=[k for k in list(res.keys())],
                        columns=[k for k in list(res.keys())])


def scatter_plot(df, df_corr):
    """
    Create a scatter_plot graphic with the DataFrame passed as an argument
    :param df: A DataFrame
    :param df_corr: A DataFrame containing Pearson's correlation coefficient
    :return: A new DataFrame containing Pearson's correlation coefficient
    """

    plt.style.use('ggplot')
    list_col = [0]
    list_name = ['', '']
    columns = list(df_corr.columns)
    index = list(df_corr.index)
    for col in columns:
        for ind in index:
            if col is not ind:
                if abs(df_corr.loc[ind, col]) > abs(list_col[0]):
                    list_col[0] = df_corr.loc[ind, col]
                    list_name = [col, ind]

    plt.scatter(df.loc[:, list_name[0]], df.loc[:, list_name[0]], label=list_name[0], c='black', s=30)
    plt.scatter(df.loc[:, list_name[1]].apply(lambda x: x * 100), df.loc[:, list_name[1]].apply(lambda x: x * 100),
                label=list_name[1], c='yellow', s=10)

    plt.legend()
    plt.tight_layout(True)


def heat_map(df_corr):
    """
    Calculates Pearson's correlation coefficients of the DataFrame passed as an argument
    and plot a heat map of correlation coefficient
    :param df: A DataFrame
    :return: A new DataFrame containing Pearson's correlation coefficient
    """

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7, 6])
    cax = ax.matshow(abs(df_corr), cmap='Reds')
    ax.set_title('Pearson\'s correlation coefficient heat map')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(range(len(df_corr.columns.values)))
    ax.set_yticks(range(len(df_corr.columns.values)))
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_xticklabels(df_corr.columns.values, rotation='vertical')
    ax.set_yticklabels(df_corr.columns.values)
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(8)
    fig.colorbar(cax)
    plt.tight_layout()

    return df_corr


def parsing():
    """
    Parses and defines parameters
    :return: _args
    """

    parser = argparse.ArgumentParser(prog='py scatter_plot.py')
    parser.add_argument('csv_file', help='A csv file containing data')
    parser.add_argument('-c', '--compare', action='store_true', help='Comparative mode', default=False)
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
            df = df.drop(columns=['Index']).dropna().reset_index(drop=True)

            df_corr = pearson_s_correlation_coefficient(df)

            if args.compare:
                df_corr_coef = heat_map(df_corr)
                print(f'\x1b[1;30;43mPandas corr():\x1b[0m \n\n{df.corr()}\n\n')
                print(f'\x1b[1;30;42mDSLR Pearson\'s correlation coefficient:\x1b[0m \n\n{df_corr}\n\n')
            else:
                scatter_plot(df, df_corr)

            if args.save:
                plt.savefig(os.path.join(os.getcwd(), 'scatter_plot.png'))

            plt.show()
        else:
            sys.exit(print(f'\x1b[1;37;41mEmpty DataFrame \x1b[0m\n'))
    else:
        sys.exit(print(f'\x1b[1;37;41mThe selected file must be a csv file \x1b[0m\n'))
