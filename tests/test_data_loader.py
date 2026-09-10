import json

import pandas as pd

from src import data_loader


def test_get_stock_redownloads_when_cache_request_changes(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(data_loader, "RAW_DATA_DIR", tmp_path)
    downloads = []

    def fake_download(ticker: str, start_date: str, end_date: str | None = None):
        downloads.append((ticker, start_date, end_date))
        data = pd.DataFrame(
            {"Close": [100.0, 101.0]},
            index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
        )
        path = data_loader.raw_data_path(ticker)
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(path)
        data_loader.raw_data_metadata_path(ticker).write_text(
            json.dumps(data_loader._cache_metadata(start_date, end_date)),
            encoding="utf-8",
        )
        return data

    monkeypatch.setattr(data_loader, "download_stock", fake_download)

    data_loader.get_stock("TEST", "2020-01-01")
    data_loader.get_stock("TEST", "2010-01-01")
    data_loader.get_stock("TEST", "2010-01-01")

    assert downloads == [
        ("TEST", "2020-01-01", data_loader.effective_end_date()),
        ("TEST", "2010-01-01", data_loader.effective_end_date()),
    ]


def test_open_ended_cache_refreshes_on_next_utc_day(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    monkeypatch.setattr(data_loader, "RAW_DATA_DIR", tmp_path)
    boundary = ["2021-01-04"]
    monkeypatch.setattr(data_loader, "effective_end_date", lambda end_date=None: end_date or boundary[0])
    calls = []
    def provider(ticker, **kwargs):
        calls.append(kwargs["end"])
        return pd.DataFrame({"Close": [100.0, 101.0]}, index=pd.to_datetime(["2021-01-01", "2021-01-04"]))
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=provider))

    first = data_loader.get_stock("TEST", "2020-01-01")
    data_loader.get_stock("TEST", "2020-01-01")
    assert calls == ["2021-01-04"]
    assert first.index.max() == pd.Timestamp("2021-01-01")
    boundary[0] = "2021-01-05"
    refreshed = data_loader.get_stock("TEST", "2020-01-01")
    assert calls == ["2021-01-04", "2021-01-05"]
    assert refreshed.index.max() == pd.Timestamp("2021-01-04")
    assert json.loads(data_loader.raw_data_metadata_path("TEST").read_text())["end_date"] == "2021-01-05"


def test_fixed_snapshot_force_download_and_broken_metadata(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    monkeypatch.setattr(data_loader, "RAW_DATA_DIR", tmp_path)
    calls = []
    def provider(ticker, **kwargs):
        calls.append(kwargs["end"])
        return pd.DataFrame({"Close": [100.0]}, index=pd.to_datetime(["2020-01-02"]))
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=provider))
    for end in ["2021-01-01", "2021-01-01", "2022-01-01"]:
        data_loader.get_stock("TEST", "2020-01-01", end)
    assert len(calls) == 2
    data_loader.get_stock("TEST", "2020-01-01", "2022-01-01", force_download=True)
    data_loader.raw_data_metadata_path("TEST").write_text("broken", encoding="utf-8")
    data_loader.get_stock("TEST", "2020-01-01", "2022-01-01")
    assert len(calls) == 4
