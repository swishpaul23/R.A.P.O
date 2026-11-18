import pandas as pd
import pytest

from rapo.data import default_cache_dir, fetch_prices, risk_free_daily, snapshot_downloader

DAYS = pd.bdate_range("2024-01-01", "2024-03-29")


class FakeDownloader:
    """Serves deterministic price series and records every call."""

    def __init__(self, series: dict[str, pd.Series]):
        self.series = series
        self.calls: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []

    def __call__(self, ticker, start, end):
        self.calls.append((ticker, pd.Timestamp(start), pd.Timestamp(end)))
        s = self.series.get(ticker)
        if s is None:
            return pd.Series(dtype=float)
        return s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]


@pytest.fixture
def fake():
    a = pd.Series(range(1, len(DAYS) + 1), index=DAYS, dtype=float)
    b = pd.Series(range(1, len(DAYS) + 1), index=DAYS, dtype=float)[2:]  # starts 2 days later
    irx = pd.Series(5.04, index=DAYS)
    return FakeDownloader({"A": a, "B": b, "^IRX": irx})


def test_fetch_prices_outer_joins_and_keeps_order(tmp_path, fake):
    df = fetch_prices(["A", "B"], "2024-01-01", "2024-01-31", tmp_path, fake)
    assert list(df.columns) == ["A", "B"]
    assert df["B"].isna().sum() == 2
    assert df.index.min() == pd.Timestamp("2024-01-01")


def test_fetch_prices_uses_cache_second_time(tmp_path, fake):
    fetch_prices(["A", "B"], "2024-01-01", "2024-01-31", tmp_path, fake)
    fetch_prices(["A", "B"], "2024-01-01", "2024-01-31", tmp_path, fake)
    assert len(fake.calls) == 2


def test_fetch_prices_downloads_only_missing_tail(tmp_path, fake):
    fetch_prices(["A"], "2024-01-01", "2024-01-10", tmp_path, fake)
    df = fetch_prices(["A"], "2024-01-01", "2024-01-20", tmp_path, fake)
    assert fake.calls[1][1] > pd.Timestamp("2024-01-10")
    assert df.index.max() == pd.Timestamp("2024-01-19")  # last business day <= 20th
    assert not df.index.duplicated().any()


def test_fetch_prices_bad_ticker_raises_with_name(tmp_path, fake):
    with pytest.raises(ValueError, match="ZZZZ"):
        fetch_prices(["A", "ZZZZ"], "2024-01-01", "2024-02-01", tmp_path, fake)


def test_risk_free_daily_converts_percent(tmp_path, fake):
    rf = risk_free_daily("2024-01-01", "2024-01-31", tmp_path, fake)
    assert len(rf) > 0
    assert rf.to_numpy() == pytest.approx(0.0504 / 252)


def test_default_cache_dir_respects_env(monkeypatch, tmp_path):
    monkeypatch.setenv("RAPO_CACHE_DIR", str(tmp_path / "c"))
    assert default_cache_dir() == tmp_path / "c"


def test_default_cache_dir_uses_tmp_on_vercel(monkeypatch):
    monkeypatch.delenv("RAPO_CACHE_DIR", raising=False)
    monkeypatch.setenv("VERCEL", "1")
    assert str(default_cache_dir()).startswith("/tmp")


def test_default_cache_dir_local(monkeypatch):
    monkeypatch.delenv("RAPO_CACHE_DIR", raising=False)
    monkeypatch.delenv("VERCEL", raising=False)
    assert str(default_cache_dir()).replace("\\", "/") == ".cache/prices"


def test_snapshot_downloader_reads_csv_and_slices(tmp_path):
    s = pd.Series([1.0, 2.0, 3.0], index=pd.bdate_range("2024-01-01", periods=3), name="price")
    s.index.name = "date"
    s.to_csv(tmp_path / "_IRX.csv")
    dl = snapshot_downloader(tmp_path)
    out = dl("^IRX", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"))
    assert out.tolist() == [2.0, 3.0]
    assert dl("NOPE", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03")).empty
