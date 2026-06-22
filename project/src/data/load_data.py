from datetime import datetime
import functools
from typing import Annotated

from dateutil.relativedelta import relativedelta
import pandas as pd
import time
import yfinance as yf

from project.src.data.synthetic import generate_ohlcv
from project.src.utils.cache import get_data_dir


def yf_retry(func, max_retries=3, base_delay=2.0):
    """Execute a yfinance call with exponential backoff on rate limits."""
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
            else:
                raise


def _normalize_ohlcv(data: pd.DataFrame, symbol: str, source: str) -> pd.DataFrame:
    df = data.copy()
    if "Date" not in df.columns:
        df = df.reset_index()
        date_col = df.columns[0]
        df = df.rename(columns={date_col: "Date"})

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if "Volume" not in df.columns:
        df["Volume"] = 0

    df.attrs["source"] = source
    df.attrs["symbol"] = symbol.upper()
    return df


@functools.lru_cache(maxsize=10)
def _get_YFin_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):
    datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    canonical = symbol.upper()
    ticker = yf.Ticker(canonical)
    yf_retry(lambda: ticker.info.get("currency", "USD"))
    end_inclusive = (end_dt + relativedelta(days=1)).strftime("%Y-%m-%d")
    data = yf_retry(lambda: ticker.history(start=start_date, end=end_inclusive))

    if data.empty:
        raise Exception(f"[{symbol}]: no rows between {start_date} and {end_date}")

    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close"]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].round(2)

    data_path = get_data_dir() / symbol / start_date / f"{end_date}.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data.reset_index().to_csv(data_path, index=False)

    return _normalize_ohlcv(data, symbol, source="yfinance")


def get_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):
    data_path = get_data_dir() / symbol / start_date / f"{end_date}.csv"

    if data_path.exists():
        df = pd.read_csv(data_path)
        return _normalize_ohlcv(df, symbol, source="cache")
    return _get_YFin_data_online(symbol, start_date, end_date)


def get_data_with_fallback(
    symbol: str,
    start_date: str,
    end_date: str,
    use_synthetic: bool = False,
) -> pd.DataFrame:
    if use_synthetic:
        return generate_ohlcv(symbol, start_date, end_date)

    try:
        return get_data(symbol, start_date, end_date)
    except Exception as exc:
        print(f"[warn] yfinance failed ({exc}), using synthetic data")
        return generate_ohlcv(symbol, start_date, end_date)
