import numpy as np
import pandas as pd
import pytest

from aem_research.data import simfin_cache as sc


def simfin_prices_frame():
    """Shape of sf.load_shareprices(variant='daily'): (Ticker, Date) MultiIndex."""
    dates = pd.bdate_range("2012-01-03", periods=4)
    rows = []
    for t, sid in (("AAPL", 111052), ("BRK-B", 2), ("GONE", 999)):
        for d in dates[: (2 if t == "GONE" else 4)]:
            rows.append({"Ticker": t, "Date": d, "SimFinId": sid, "Open": 10.0, "High": 11.0,
                         "Low": 9.0, "Close": 10.5, "Adj. Close": 10.2, "Volume": 1e6,
                         "Dividend": np.nan, "Shares Outstanding": 5e8})
    return pd.DataFrame(rows).set_index(["Ticker", "Date"])


def simfin_companies_frame():
    return pd.DataFrame({"Ticker": ["AAPL", "BRK-B", "GONE"], "SimFinId": [111052, 2, 999],
                         "Company Name": ["APPLE INC", "BERKSHIRE", "GONE CORP"],
                         "CIK": [320193.0, 1067983.0, np.nan]}).set_index("Ticker")


def test_normalise_promotes_index_and_renames():
    df = sc.normalise(simfin_prices_frame())
    for col in ("ticker", "date", "simfin_id", "adj_close", "shares_out", "close"):
        assert col in df.columns


def test_load_from_pickle_cache_like_cvm_raw(tmp_path):
    raw = tmp_path / "cvm-research" / "data" / "raw"
    raw.mkdir(parents=True)
    simfin_prices_frame().to_pickle(raw / "us_prices.pkl")
    simfin_companies_frame().to_pickle(raw / "us_companies.pkl")
    cache = sc.load([raw, tmp_path / "cvm-research" / "data"])
    assert cache.found
    p = cache.prices
    assert set(p["ticker"].astype(str)) == {"AAPL", "BRK.B", "GONE"}
    assert str(p["date"].dtype) == "datetime64[ns]" and str(p["adj_close"].dtype) == "float32"
    assert cache.companies["cik"].tolist()[:2] == [320193, 1067983]
    assert pd.isna(cache.companies["cik"].iloc[2])
    assert sc.best_price_column(p) == "adj_close"


def test_load_from_simfin_csv_cache(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    simfin_prices_frame().reset_index().to_csv(data / "us-shareprices-daily.csv", sep=";", index=False)
    simfin_companies_frame().reset_index().to_csv(data / "us-companies.csv", sep=";", index=False)
    cache = sc.load([tmp_path / "missing", data])
    assert cache.found and len(cache.prices) == 10
    assert cache.files["prices"].endswith("us-shareprices-daily.csv")


def test_prices_without_ticker_are_joined_via_simfin_id(tmp_path):
    prices = simfin_prices_frame().reset_index().drop(columns="Ticker")
    prices.to_pickle(tmp_path / "us_prices.pkl")
    simfin_companies_frame().to_pickle(tmp_path / "us_companies.pkl")
    cache = sc.load([tmp_path])
    assert cache.found and "joined via SimFinId" in " ".join(cache.notes)


def test_reused_ticker_is_split_by_simfin_id(tmp_path):
    frame = simfin_prices_frame().reset_index()
    reuse = frame[frame["Ticker"] == "GONE"].assign(SimFinId=12345,
                                                     Date=pd.bdate_range("2020-01-02", periods=2).repeat(1))
    pd.concat([frame, reuse]).set_index(["Ticker", "Date"]).to_pickle(tmp_path / "us_prices.pkl")
    comp = simfin_companies_frame().reset_index()
    comp = pd.concat([comp, pd.DataFrame({"Ticker": ["GONE"], "SimFinId": [12345], "Company Name": ["NEW GONE"],
                                          "CIK": [555.0]})]).set_index("Ticker")
    comp.to_pickle(tmp_path / "us_companies.pkl")
    cache = sc.load([tmp_path])
    labels = set(cache.prices["ticker"].astype(str))
    assert {"GONE~999", "GONE~12345", "AAPL", "BRK.B"} == labels
    assert set(cache.companies["ticker"]) == {"AAPL", "BRK.B", "GONE~999", "GONE~12345"}
    assert any("more than one SimFin company" in n for n in cache.notes)


def test_missing_cache_is_reported(tmp_path):
    cache = sc.load([tmp_path])
    assert not cache.found and cache.files == {"prices": None, "companies": None}
    assert any("no US share-price file" in n for n in cache.notes)


def test_parquet_roundtrip_keeps_multiindex(tmp_path):
    pytest.importorskip("pyarrow")
    simfin_prices_frame().to_parquet(tmp_path / "us_prices.parquet")
    cache = sc.load([tmp_path])
    assert cache.found and len(cache.prices) == 10
