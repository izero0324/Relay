"""
universe.py — Fetch and deduplicate the investable universe.

Each source tries multiple methods in priority order, falling back
gracefully so you always get a universe even if one endpoint is down.

S&P 500 fetch order:
  1. Wikipedia (with browser User-Agent to avoid 403)
  2. GitHub-hosted CSV (datasets/s-and-p-500-companies)

NASDAQ-100 fetch order:
  1. Wikipedia (with browser User-Agent)
  2. Slickcharts HTML table

Add more sources by extending _FETCHERS at the bottom.
"""

import io
import logging

import pandas as pd
import requests

from config import UNIVERSE_SOURCES

logger = logging.getLogger(__name__)

# ── URLs ──────────────────────────────────────────────────────────────────────
_SP500_WIKI_URL  = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# The constituents table lives on its own article (the main Nasdaq-100
# page no longer carries it). URL is case-sensitive: "NASDAQ", not "Nasdaq".
_NDX100_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"

# Free GitHub-hosted CSV — maintained community dataset
_SP500_GH_URL  = (
    "https://raw.githubusercontent.com/datasets/"
    "s-and-p-500-companies/main/data/constituents.csv"
)
_NDX100_SLICKCHARTS_URL = "https://www.slickcharts.com/nasdaq100"

# Browser User-Agent — avoids Wikipedia / Cloudflare 403s
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _clean(ticker: str) -> str:
    """Yahoo Finance uses '-' not '.' — e.g. BRK.B → BRK-B."""
    return str(ticker).strip().replace(".", "-")


def _html_to_tables(url: str) -> list[pd.DataFrame]:
    """Fetch a URL with a browser User-Agent and parse its HTML tables."""
    resp = requests.get(url, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))


def _csv_from_url(url: str) -> pd.DataFrame:
    """Download a raw CSV via requests (also sends browser headers)."""
    resp = requests.get(url, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


# ── S&P 500 ───────────────────────────────────────────────────────────────────

def _sp500_from_wikipedia() -> list[str]:
    tables = _html_to_tables(_SP500_WIKI_URL)
    df = tables[0]
    col = next((c for c in df.columns if "symbol" in str(c).lower()), None)
    if col is None:
        raise ValueError(f"No Symbol column found. Columns: {df.columns.tolist()}")
    return [_clean(t) for t in df[col].dropna()]


def _sp500_from_github() -> list[str]:
    df = _csv_from_url(_SP500_GH_URL)
    col = next((c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()), None)
    if col is None:
        raise ValueError(f"No ticker column found. Columns: {df.columns.tolist()}")
    return [_clean(t) for t in df[col].dropna()]


def _fetch_sp500() -> list[str]:
    sources = [
        ("Wikipedia",  _sp500_from_wikipedia),
        ("GitHub CSV", _sp500_from_github),
    ]
    for name, fn in sources:
        try:
            tickers = fn()
            if len(tickers) >= 400:          # sanity check
                logger.info(f"S&P 500 ({name}): {len(tickers)} tickers")
                return tickers
            logger.warning(f"S&P 500 ({name}): only {len(tickers)} tickers — trying next source")
        except Exception as e:
            logger.warning(f"S&P 500 ({name}) failed: {e}")
    logger.error("S&P 500: all sources failed")
    return []


# ── NASDAQ-100 ────────────────────────────────────────────────────────────────

def _ndx100_from_wikipedia() -> list[str]:
    tables = _html_to_tables(_NDX100_WIKI_URL)
    for df in tables:
        # str(c): some page tables have integer column labels
        cols_lower = [str(c).lower() for c in df.columns]
        if "ticker" in cols_lower:
            col = df.columns[cols_lower.index("ticker")]
            tickers = [_clean(t) for t in df[col].dropna()]
            if len(tickers) >= 90:
                return tickers
        if "symbol" in cols_lower:
            col = df.columns[cols_lower.index("symbol")]
            tickers = [_clean(t) for t in df[col].dropna()]
            if len(tickers) >= 90:
                return tickers
    raise ValueError("Could not find a NASDAQ-100 ticker column on Wikipedia")


def _ndx100_from_slickcharts() -> list[str]:
    tables = _html_to_tables(_NDX100_SLICKCHARTS_URL)
    for df in tables:
        cols_lower = [str(c).lower() for c in df.columns]
        if "symbol" in cols_lower:
            col = df.columns[cols_lower.index("symbol")]
            tickers = [_clean(t) for t in df[col].dropna()]
            if len(tickers) >= 90:
                return tickers
    raise ValueError("Could not find a NASDAQ-100 symbol column on Slickcharts")


def _fetch_nasdaq100() -> list[str]:
    sources = [
        ("Wikipedia",   _ndx100_from_wikipedia),
        ("Slickcharts", _ndx100_from_slickcharts),
    ]
    for name, fn in sources:
        try:
            tickers = fn()
            if len(tickers) >= 90:
                logger.info(f"NASDAQ-100 ({name}): {len(tickers)} tickers")
                return tickers
            logger.warning(f"NASDAQ-100 ({name}): only {len(tickers)} tickers — trying next")
        except Exception as e:
            logger.warning(f"NASDAQ-100 ({name}) failed: {e}")
    logger.error("NASDAQ-100: all sources failed")
    return []


# ── Registry ──────────────────────────────────────────────────────────────────

_FETCHERS = {
    "sp500":     _fetch_sp500,
    "nasdaq100": _fetch_nasdaq100,
}


def get_universe() -> list[str]:
    """
    Return a deduplicated list of tickers from all configured sources.
    Deduplication preserves first-occurrence order.
    """
    raw: list[str] = []
    for source in UNIVERSE_SOURCES:
        fetcher = _FETCHERS.get(source)
        if fetcher is None:
            logger.warning(f"Unknown universe source: '{source}' — skipping")
            continue
        raw.extend(fetcher())

    seen: set[str] = set()
    unique: list[str] = []
    for t in raw:
        if t and t not in seen:
            seen.add(t)
            unique.append(t)

    logger.info(f"Universe total: {len(unique)} unique tickers")
    return unique
