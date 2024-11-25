import requests
import json

def get_all_tickers():
    """
    Fetches the list of all tickers from the issuer directory.

    Returns:
        list: A list of tickers, or an empty list if the request fails.
    """
    url = 'http://10.192.1.245:8080/issuer-directory/list'
    headers = {'Content-Type': 'application/json'}
    data = {"keywords": ""}

    items = []
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        items = response.json().get('data', [])  # Assuming 'tickers' is the key in the response
    except requests.RequestException as e:
        print(f"Error fetching tickers: {e}")
        return []
    tickers = [i["secCode"] for i in items]
    return tickers

def get_tic_data_in_range(tic: str, start_date: str, end_date: str):
    """
    Fetches trading data for a specific ticker within a date range.

    Args:
        tic (str): The ticker symbol.
        start_date (str): The start date in YYYY-MM-DD format.
        end_date (str): The end date in YYYY-MM-DD format.

    Returns:
        dict: Trading data for the ticker in the given date range, or an empty dict if the request fails.
    """
    url = 'http://10.192.1.245:8080/orderbook/basic-trading-data'
    headers = {'Content-Type': 'application/json'}
    data = {
        "secCode": tic,
        "startDate": start_date,
        "endDate": end_date
    }

    items = []
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        items = response.json().get("data", [])
    except requests.RequestException as e:
        print(f"Error fetching trading data for {tic}: {e}")
        return []
    return items

def get_latest_tic_data(tic: str):
    """
    Fetches the latest OHLCV data for a specific ticker.

    Args:
        tic (str): The ticker symbol.

    Returns:
        dict: Latest OHLCV data for the ticker, or an empty dict if the request fails.
    """
    url = 'http://10.192.1.245:8080/orderbook/header'
    headers = {'Content-Type': 'application/json'}
    data = {"secCode": tic}

    result = {}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json().get("data", {})
    except requests.RequestException as e:
        print(f"Error fetching latest data for {tic}: {e}")
        return {}
    return result

