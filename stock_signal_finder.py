import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file, sheet):
    df = pd.ExcelFile(file).parse(sheet)
    df = df.set_index(df.columns[0])
    # Linear interpolation for each stock to fill NAs
    df = df.interpolate(method='linear', limit_direction="forward", axis=0)
    return df


def prices_to_returns(df, upper_bound, lower_bound):
    # Assumes rows are dates, cols are tickers
    # print(df.iloc[:20, 6:20])
    return_df = df.pct_change()
    # print(return_df.iloc[:10, 14:24])
    # Adjust any daily return beyond outlier bounds
    return_df[return_df > upper_bound] = upper_bound
    return_df[return_df < lower_bound] = lower_bound
    return return_df


def get_average_signal(price_df, return_df, date, before, after, use_ma, window_size=0):
    # Get valid tickers (no NAs in selected range around date)
    date_list = list(return_df.index)
    date_index = date_list.index(date)
    row_before = return_df.iloc[date_index - before, :]
    row_date = return_df.iloc[date_index, :]
    na_before = [i for i in range(len(row_before)) if np.isnan(row_before[i])]
    row_after = return_df.iloc[date_index + after, :]
    na_after = [i for i in range(len(row_after)) if np.isnan(row_after[i])]
    na_before.extend(na_after)
    na = list(set(na_before))
    valid_tickers = [e for i, e in enumerate(return_df.columns) if i not in na]
    #print(valid_tickers)

    # Get returns from date to date+after
    valid_price_df = price_df.loc[:, valid_tickers]
    prow_before = valid_price_df.iloc[date_index - before - 1, :]
    prow_date = valid_price_df.iloc[date_index, :]
    prow_after = valid_price_df.iloc[date_index + after, :]
    valid_price_df = valid_price_df.iloc[date_index - before - 1:date_index + after, :]
    forward_returns = [j / i-1 for i, j in zip(prow_date, prow_after)]
    backward_returns = [j / i - 1 for i, j in zip(prow_before, prow_date)]

    # Moving average return
    if use_ma:
        for ticker in valid_tickers:
            return_df.loc[:, ticker] = return_df.loc[:, ticker].rolling(window=window_size).mean()

    # Filter return df to "valid" tickers (i.e. no NAs)
    valid_return_df = return_df.loc[:, valid_tickers]
    valid_return_df = valid_return_df.iloc[date_index - before:date_index + after, :]

    # Separate stocks by positive vs negative forward returns
    pos_tickers = [e for i, e in enumerate(valid_return_df.columns) if forward_returns[i] > 0]
    pos_returns = [forward_returns[i] for i, e in enumerate(valid_return_df.columns) if forward_returns[i] > 0]
    pos_weights = np.divide(pos_returns, sum(pos_returns))
    pos_returns = [n / max(pos_returns) for n in pos_returns]
    neg_tickers = [e for i, e in enumerate(valid_return_df.columns) if forward_returns[i] <= 0]
    neg_returns = [forward_returns[i] for i, e in enumerate(valid_return_df.columns) if forward_returns[i] <= 0]
    neg_weights = np.divide(neg_returns, sum(neg_returns))
    neg_returns = [-n / min(neg_returns) for n in neg_returns]
    pos_return_df = valid_return_df.loc[:, pos_tickers]
    neg_return_df = valid_return_df.loc[:, neg_tickers]
    return_df = pd.concat([pos_return_df, neg_return_df], axis=1)

    # Determine colors based on returns
    pos_colors = [128 * i + 128 for i in pos_returns]
    neg_colors = [128 * i + 128 for i in neg_returns]
    weighted_colors = pos_colors
    weighted_colors.extend(neg_colors)

    # Weight stock returns based on eventual forward return
    weighted_pos_return_df = pos_return_df.mul(pos_weights, axis=1)
    weighted_neg_return_df = neg_return_df.mul(neg_weights, axis=1)
    weighted_return_df = pd.concat([weighted_pos_return_df, weighted_neg_return_df], axis=1)

    # Average return signal = sum of returns (weighted by forward return)
    pos_return_signal = weighted_pos_return_df.sum(axis=1)
    neg_return_signal = weighted_neg_return_df.sum(axis=1)

    return pos_return_signal, neg_return_signal, pos_return_df, neg_return_df, pos_colors, neg_colors




file = 'C:/Users/88chr/Documents/Healthcare Stock Analysis/Tickers for Google Finance.xlsx'
sheet = 'Dated Prices'
#file = 'C:/Users/88chr/Documents/Healthcare Stock Analysis/Toy Set.xlsx'
#sheet = 'Toy Set'
price_df = load_data(file, sheet)

# Manually drop columns of weird stocks
drop_list = ['RIOT']
price_df = price_df.drop(drop_list, axis=1)

return_df = prices_to_returns(price_df, 0.6, -0.6)
return_df = return_df.iloc[1:]
save_name = 'C:/Users/88chr/Documents/Interpolated Returns.csv'
return_df.to_csv(save_name)

date = 44104
pos_signal, neg_signal, pos_return_df, neg_return_df, pos_colors, neg_colors = get_average_signal(price_df, return_df, date, 22, 22, True, 5)

signal_df = pd.DataFrame()
signal_df['Pos'] = pos_signal
signal_df['Neg'] = neg_signal
save_name = 'C:/Users/88chr/Documents/Stock Signals.csv'
signal_df.to_csv(save_name)

cmap = matplotlib.cm.get_cmap('RdYlGn') #cmap goes from 0 to 256

fig, axes = plt.subplots(nrows=2, ncols=2)

pos_color_values = [int(n) for n in pos_colors]
pos_colors = [cmap(v) for v in pos_color_values]
pos_return_df.plot(color=pos_colors, ax=axes[0, 0])
axes[0, 0].legend(loc='upper left', fontsize=6)

neg_color_values = [int(n) for n in neg_colors]
neg_colors = [cmap(v) for v in neg_color_values]
neg_return_df.plot(color=neg_colors, ax=axes[0, 1])
axes[0, 1].legend(loc='upper left', fontsize=6)

pos_signal.plot(ax=axes[1,0])
axes[1,0].vlines(x=date, ymin=min(pos_signal), ymax=max(pos_signal), colors='r')
neg_signal.plot(ax=axes[1,1])
axes[1,1].vlines(x=date, ymin=min(neg_signal), ymax=max(neg_signal), colors='r')

plt.show()