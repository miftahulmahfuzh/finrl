from utils import get_all_tickers, get_tic_data_in_range, get_latest_tic_data

def test_get_all_tickers():
    print("Testing get_all_tickers...")
    tickers = get_all_tickers()
    print("Tickers:", tickers)
    print(f"Total tickers: {len(tickers)}")

def test_get_tic_data_in_range():
    print("Testing get_tic_data_in_range...")
    ticker = "BBCA"
    start_date = "2024-11-22"
    end_date = "2024-11-22"
    data = get_tic_data_in_range(ticker, start_date, end_date)
    print(f"Trading data for {ticker} from {start_date} to {end_date}:", data)

def test_get_latest_tic_data():
    print("Testing get_latest_tic_data...")
    ticker = "BBCA"
    latest_data = get_latest_tic_data(ticker)
    print(f"Latest data for {ticker}:", latest_data)

if __name__ == "__main__":
    test_get_all_tickers()
    print("\n")
    test_get_tic_data_in_range()
    print("\n")
    test_get_latest_tic_data()

