import pandas as pd

def load_stock_data(csv_path):
    data = pd.read_csv(csv_path)

    # Rename DateTime → Date
    data.rename(columns={
        "DateTime": "Date",
        "close": "Close"
    }, inplace=True)

    # Let pandas auto-parse (ISO format)
    data["Date"] = pd.to_datetime(data["Date"])

    # Set Date as index
    data.set_index("Date", inplace=True)

    if data.empty:
        raise ValueError("CSV loaded but contains no data")

    return data 