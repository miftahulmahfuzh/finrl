import pandas as pd

d = "/mnt/c/Users/mahfu/Downloads/tuntun/tuntun_ubuntu/github_me/finrl/from_finrl-tutorials_git/data_1993_to_2024/data"
t = f"{d}/75_tic_price_variation_v3-a_train_12_features.csv"
df = pd.read_csv(t)
t = f"{d}/75_tic_price_variation_v3-a_train_12_features.xlsx"
df.to_excel(t, sheet_name="rows", index=False)
print(f"data is saved to {t}")
