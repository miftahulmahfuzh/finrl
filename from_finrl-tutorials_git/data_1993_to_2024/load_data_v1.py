import pandas as pd
fname = "/home/devmiftahul/trading_model/from_finrl-tutorials_git/processed_by_gavin/preprocessed_data.csv"
df = pd.read_csv(fname)

# TICKERS
# tickers = sorted(df["tick"].unique())
# print(len(tickers))
# tickers_str = "\n".join(tickers)
# with open("ticker_1120.txt", "w+") as f:
#    f.write(tickers_str)

# DATE
dates = sorted(df["date"].unique())
dates_str = "\n".join(dates)
with open("dates_1120_tickers.txt", "w+") as f:
    f.write(dates_str)
