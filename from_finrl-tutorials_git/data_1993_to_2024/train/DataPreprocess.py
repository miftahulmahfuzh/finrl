import pandas as pd

df = pd.read_excel('75_tuntun_api_data_with_features_v3-a.xlsx')

df_new = df[['date','tic','close','high','low','volume']].copy()
print(len(df_new))

stocks = df_new['tic'].unique()
print(len(stocks))

window = 30

df_new['close_n'] = -1
df_new['high_n'] = -1
df_new['low_n'] = -1
df_new['volume_n'] = -1


def normalize_window(group):
    for col in ['close', 'high', 'low', 'volume']:
        group[f'{col}_max'] = group[col].rolling(window, min_periods=1).max()
        group[f'{col}_min'] = group[col].rolling(window, min_periods=1).min()

        group[f'{col}_n'] = (group[col] - group[f'{col}_min']) / (
                group[f'{col}_max'] - group[f'{col}_min']
        )
        group[f'{col}_n'] = group[f'{col}_n'].fillna(0.5)
    return group


df_new = df_new.groupby('tic', group_keys=False).apply(normalize_window)

df_new.to_csv('output.csv')