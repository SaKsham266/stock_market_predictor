import pandas as pd


def adjust_for_splits(data, close_col="Close", threshold=0.3):
    """
    Detect unadjusted stock splits/bonus issues (a single-day price move
    larger than `threshold`, e.g. a 2:1 split roughly halving the price)
    and back-adjust all prices before each split so the series is
    continuous. Without this, a split shows up as a fake ~-50% "return"
    that corrupts Return/MA_10/Volatility_10 and destabilizes training.
    """
    data = data.copy()
    pct = data[close_col].pct_change()
    split_positions = [i for i, v in enumerate(pct.abs() > threshold) if v]

    for pos in split_positions:
        ratio = data[close_col].iloc[pos - 1] / data[close_col].iloc[pos]
        ratio = round(ratio)  # snap to nearest whole-number split ratio (2, 3, ...)
        if ratio <= 1:
            continue  # not an actual split, skip (safety guard)
        data.iloc[:pos, data.columns.get_loc(close_col)] = (
            data[close_col].iloc[:pos] / ratio
        )

    return data


def load_stock_data(csv_path):
    data = pd.read_csv(csv_path)

    # Rename DateTime → Date
    data.rename(columns={
        "DateTime": "Date",
        "close": "Close"
    }, inplace=True)

    # Let pandas auto-parse (ISO format)
    data["Date"] = pd.to_datetime(data["Date"])

    # Sort chronologically — adjust_for_splits assumes ascending date order
    data = data.sort_values("Date")

    # Set Date as index
    data.set_index("Date", inplace=True)

    if data.empty:
        raise ValueError("CSV loaded but contains no data")

    data = adjust_for_splits(data)

    return data