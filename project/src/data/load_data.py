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


def attach_vix(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    try:
        from datetime import datetime
        from dateutil.relativedelta import relativedelta
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        end_inclusive = (end_dt + relativedelta(days=1)).strftime("%Y-%m-%d")

        vix_raw = yf_retry(
            lambda: yf.download(
                "^VIX",
                start=start_date,
                end=end_inclusive,
                progress=False,
                auto_adjust=True,
            )
        )

        if vix_raw.empty:
            print("[warn] VIX data empty, skipping vix column")
            return df

        if isinstance(vix_raw.columns, pd.MultiIndex):
            vix_raw.columns = vix_raw.columns.get_level_values(0)

        vix_series = vix_raw[["Close"]].copy()
        vix_series.columns = ["vix"]
        vix_series = vix_series.reset_index()
        vix_series.rename(columns={vix_series.columns[0]: "Date"}, inplace=True)
        vix_series["Date"] = pd.to_datetime(vix_series["Date"])

        if vix_series["Date"].dt.tz is not None:
            vix_series["Date"] = vix_series["Date"].dt.tz_localize(None)

        out = df.merge(vix_series[["Date", "vix"]], on="Date", how="left")
        print(f"[info] VIX attached: {out['vix'].notna().sum()} rows with data")
        return out

    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[warn] VIX fetch failed ({exc}), skipping vix column")
        return df


def get_data_with_fallback(
    symbol: str,
    start_date: str,
    end_date: str,
    use_synthetic: bool = False,
) -> pd.DataFrame:
    if use_synthetic:
        df = generate_ohlcv(symbol, start_date, end_date)
        return attach_vix(df, start_date, end_date)

    try:
        df = get_data(symbol, start_date, end_date)
    except Exception as exc:
        print(f"[warn] yfinance failed ({exc}), using synthetic data")
        df = generate_ohlcv(symbol, start_date, end_date)

    return attach_vix(df, start_date, end_date)