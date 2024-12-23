import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from utils import get_all_tickers, get_tic_data_in_range

# Constants
# START_DATE = "2010-01-01"
# END_DATE = "2024-11-26"
START_DATE = "2024-03-13"
END_DATE = "2024-11-26"
CSV_DIR = "csv_23122024"

def get_day_name(date):
    """Returns the day name of the given date."""
    day_name = datetime.strptime(date, "%Y-%m-%d").strftime("%A")
    return day_name

def daterange(start_date, end_date):
    """Generate a range of dates from start_date to end_date."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    for n in range((end - start).days + 1):
        yield (start + timedelta(n)).strftime("%Y-%m-%d")

def fetch_data_batch():
    global RESET
    # Initialize variables
    if not os.path.exists(CSV_DIR):
        os.makedirs(CSV_DIR, exist_ok=True)

    # if os.path.exists(FILENAME) and not RESET:
    #     existing_data = pd.read_csv(FILENAME)
    #     processed_days = set(existing_data["day"])
    # else:
    #     processed_days = set()
    #     existing_data = pd.DataFrame()

    # all_tickers = get_all_tickers()
    # # skip_tickers = ["BOAT", "DAAZ"]
    skip_tickers = []
    # skip_tickers = open("skip_tickers.txt").read().splitlines()
    # all_tickers = [t for t in all_tickers if t not in skip_tickers]
    # all_tickers = ["BBCA"]
    tic75 = open("75_tickers.txt").read().splitlines()
    all_tickers = tic75
    print(f"TOTAL TICKERS: {len(all_tickers)}")

    total_days = (datetime.strptime(END_DATE, "%Y-%m-%d") - datetime.strptime(START_DATE, "%Y-%m-%d")).days + 1

    # Iterate over each date and fetch data
    for tic in all_tickers:
        # if i in processed_days:
        #     continue
        RESET = True

        # print(f"Processing data for: {current_date}")
        # rows = []  # Collect rows for the current date
        # for tic in tqdm(all_tickers, total=len(all_tickers)):
        print(f"Processing {tic}..")
        # for i, current_date in tqdm(enumerate(daterange(START_DATE, END_DATE), start=1), total=total_days):
        #     day_name = get_day_name(current_date)
        #     if day_name in ["Saturday", "Sunday"]:
        #         continue

        #     data = {}
        #     try:
        #         data = get_tic_data_in_range(tic, current_date, current_date)[0]
        #     except:
        #         continue
        #         # print(f"Failed to get {tic} on date {current_date}. day: {day_name}")
        #         # skip_tickers.append(tic)
        #         # return None
        #     if not data:
        #         continue
        list_data = []
        try:
            list_data = get_tic_data_in_range(tic, START_DATE, END_DATE)
        except:
            # continue
            # print(f"Failed to get {tic} on date {current_date}. day: {day_name}")
            skip_tickers.append(tic)
            # return None
        # if not list_data:
        #     continue
        assert(len(list_data) > 0)

        for i, data in tqdm(enumerate(list_data, start=1), total=len(list_data)):

            row = {
                "date": data.get("transactionDate"),
                "open": data.get("openPrice"),
                "high": data.get("highPrice"),
                "low": data.get("lowPrice"),
                "close": data.get("closePrice"),
                "volume": data.get("volume"),
                "tic": tic,
                "day": i
            }
            # print(row)
            # rows.append(row)
            FILENAME = f"{CSV_DIR}/{tic}_api_data_{START_DATE}_{END_DATE}.csv"
            new_data = pd.DataFrame([row])
            # if (existing_data.empty and not os.path.exists(FILENAME)) or RESET:
            if (not os.path.exists(FILENAME)) or RESET:
                RESET = False
                new_data.to_csv(FILENAME, index=False)
            else:
                new_data.to_csv(FILENAME, mode='a', header=not os.path.exists(FILENAME), index=False)

        # Convert rows to DataFrame and append to CSV
        # if rows:
    print(f"TOTAL TICKERS: {len(all_tickers)}")
    with open("skip_labels_23122024.txt", "w+") as f:
        f.write("\n".join(skip_tickers))

if __name__ == "__main__":
    fetch_data_batch()
