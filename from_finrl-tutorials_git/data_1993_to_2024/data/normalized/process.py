import pandas as pd

mode = "dev"
d = "/mnt/c/Users/mahfu/Downloads/tuntun/tuntun_ubuntu/github_me/finrl/from_finrl-tutorials_git/data_1993_to_2024/data/normalized"
fname = f"{d}/75_tic_v3-a_{mode}_12_features_n_by_previous_time.xlsx"

columns = ["date", "tic", "close", "high", "low", "macd", "boll_ub", "boll_lb", "rsi_30", "close_30_sma"]
df = pd.read_excel(fname, sheet_name="dev")
df = df[columns]
df = df.sort_values(by="date")

print(df)
fname2 = f"{d}/75_tic_v3-a_{mode}_12_features_n_by_previous_time_edited.xlsx"
df.to_excel(fname2, sheet_name=mode, index=False)
print(f"NORMALIZATION DATA IS SAVED TO:\n{fname2}")


