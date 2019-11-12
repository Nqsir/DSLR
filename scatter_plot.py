import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from describe import describe


def scatter_plot(df):
    """
    Plots a scatter chart from a DataFrame based on metrics (such as Std, mean quartiles...)
    :param df: A DataFrame containing metrics
    """

    try:
        df = df.drop(columns=['Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand', 'Index'])\
            .dropna().reset_index(drop=True)
    except KeyError:
        pass

    columns = list(df)
    print(columns)
    col_length = len(df)
    df_described = describe(df)
    # -------------- TEST
    print(df_described.loc['Mean', 'Astronomy'])
    print(df.cov())
    for col_name in columns:
        if df[col_name].dtypes == 'int64' or df[col_name].dtypes == 'float64':
            for col in columns:
                print(f'{col_name} -------- {col}')
                if df[col].dtypes == 'int64' or df[col].dtypes == 'float64':
                    # print(df_described.loc['Std', col_name] * df_described.loc['Std', col])
                    i = 0
                    j = 0
                    sum_ = 0
                    while i < df_described.loc['Count', col_name] and j < df_described.loc['Count', col]:
                        sum_ += (df.loc[i, col_name] - (1 - df_described.loc['Mean', col_name])) *\
                                (df.loc[j, col] - (1 - df_described.loc['Mean', col]))
                        i += 1
                        j += 1
                    print(f'sum_ = {sum_}')
                    # one = sum([abs(val - df_described.loc['Mean', col_name]) for val in df[col_name]])
                    # two = sum([abs(val - df_described.loc['Mean', col]) for val in df[col]])
                    # print(one)
                    # print(two)
                    # print((one * two))
    #                 # Sum of Squares due to Regression (SSR): Tot((each predicted value - price_mean)²)
    #                 # SSR = sum([(prediction[i] - p_mean) ** 2 for i in range(nbr_data)])
    #                 covariance =
    #                 # Error Sum of Squares (SSE): Tot((each price - each predicted value)²)
    #                 # SSE = sum([(price[i] - prediction[i]) ** 2 for i in range(nbr_data)])
    #
    #                 # R2 = SSR / (SSR + SSE)
    #                 print(R2)

    # TEST END--------------

    print(df.corr())
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[5.6, 5])
    cax = ax.matshow(abs(df.corr()), cmap='Reds')
    ax.set_title('Correlation coefficients')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(range(len(df.columns.values)))
    ax.set_yticks(range(len(df.columns.values)))
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_xticklabels(df.columns.values, rotation='vertical')
    ax.set_yticklabels(df.columns.values)
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(8)
    fig.colorbar(cax)
    plt.tight_layout()


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
        # df_described = describe(df)
        scatter_plot(df)

        if args.save:
            plt.savefig(os.path.join(os.getcwd(), 'scatter_plot.png'))

        plt.show()
    else:
        sys.exit(print(f'\x1b[1;37;41mThe selected file must be a csv file \x1b[0m\n'))
