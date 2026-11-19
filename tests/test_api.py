import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import create_app

IDX = pd.bdate_range("2018-01-01", "2023-12-29")


def _walk(seed, mu, sigma):
    r = np.random.default_rng(seed).normal(mu, sigma, len(IDX))
    return pd.Series(100 * np.cumprod(1 + r), index=IDX)


UNIVERSE = {
    "SPY": _walk(1, 0.0004, 0.011),
    "TLT": _walk(2, 0.0001, 0.008),
    "GLD": _walk(3, 0.0002, 0.009),
    "FLAT": pd.Series(100.0, index=IDX),
    "LATE": _walk(4, 0.0005, 0.02).loc["2023-06-01":],
}


def fake_prices(tickers, start, end):
    missing = [t for t in tickers if t not in UNIVERSE]
    if missing:
        raise ValueError(f"No price data for: {', '.join(missing)}. Check the ticker symbols.")
    df = pd.DataFrame({t: UNIVERSE[t] for t in tickers})
    return df.loc[pd.Timestamp(start):pd.Timestamp(end)]


def fake_rf(start, end):
    return pd.Series(0.0001, index=IDX).loc[pd.Timestamp(start):pd.Timestamp(end)]


@pytest.fixture
def client():
    return TestClient(create_app(prices_loader=fake_prices, rf_loader=fake_rf))


BODY = {"tickers": ["TLT", "GLD", "SPY"], "benchmark": "SPY", "start": "2018-01-01",
        "end": "2023-12-29", "lookback": 126, "rebalance_every": 21, "cost_bps": 10}


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_backtest_shape(client):
    res = client.post("/api/backtest", json=BODY)
    assert res.status_code == 200
    data = res.json()
    names = [row["name"] for row in data["table"]]
    assert names[:5] == ["Equal weight", "Inverse vol", "Min variance", "Max Sharpe", "HRP"]
    assert names[5] == data["benchmark_label"] == "SPY (benchmark)"
    n = len(data["series"]["dates"])
    assert n > 0
    for name in names:
        assert len(data["series"]["equity"][name]) == n
        assert len(data["series"]["drawdown"][name]) == n
        assert max(data["series"]["drawdown"][name]) <= 0
    assert {s["window"] for s in data["stress"]} == {"2020 COVID crash", "2022 rate shock"}
    assert len(data["frontier"]["points"]) == 3000
    assert {s["name"] for s in data["frontier"]["strategies"]} == set(names[:5])
    for w in data["weights"].values():
        assert sum(w.values()) == pytest.approx(1.0)
    assert data["correlation"]["tickers"] == ["TLT", "GLD", "SPY"]
    assert sorted(data["clusters"]["order"]) == ["GLD", "SPY", "TLT"]


def test_benchmark_can_also_be_held(client):
    data = client.post("/api/backtest", json=BODY).json()
    assert "SPY" in data["weights"]["Equal weight"]
    assert data["weights"]["Equal weight"]["SPY"] == pytest.approx(1 / 3)


def test_bad_ticker_returns_400_naming_it(client):
    res = client.post("/api/backtest", json={**BODY, "tickers": ["SPY", "ZZZZ"]})
    assert res.status_code == 400
    assert "ZZZZ" in res.json()["detail"]


def test_window_too_short_returns_400(client):
    res = client.post("/api/backtest", json={**BODY, "start": "2023-06-01", "lookback": 252})
    assert res.status_code == 400
    assert "253" in res.json()["detail"]


def test_empty_tickers_rejected(client):
    assert client.post("/api/backtest", json={**BODY, "tickers": []}).status_code == 422


def test_tickers_are_normalized(client):
    res = client.post("/api/backtest", json={**BODY, "tickers": [" tlt", "gld ", "GLD", "spy"]})
    assert res.status_code == 200
    assert res.json()["meta"]["tickers"] == ["TLT", "GLD", "SPY"]


def test_nan_metrics_serialize_as_null(client):
    res = client.post("/api/backtest", json={**BODY, "benchmark": "FLAT"})
    assert res.status_code == 200
    bench = res.json()["table"][-1]
    assert bench["sharpe"] is None


def test_frontier_strategy_points_include_sharpe(client):
    data = client.post("/api/backtest", json=BODY).json()
    rf_annual = 0.0001 * 252
    for p in data["frontier"]["strategies"]:
        assert p["sharpe"] == pytest.approx((p["ret"] - rf_annual) / p["vol"])


def _fail_prices(tickers, start, end):
    raise ValueError(f"No price data for: {', '.join(tickers)}. Check the symbol and your internet connection.")


def _fail_rf(start, end):
    raise ValueError("No price data for: ^IRX.")


def test_live_data_reported_as_live(client):
    assert client.post("/api/backtest", json=BODY).json()["meta"]["data_source"] == "live"


def test_falls_back_to_snapshot_when_live_data_fails():
    app = create_app(prices_loader=_fail_prices, rf_loader=_fail_rf, fallback=(fake_prices, fake_rf))
    res = TestClient(app).post("/api/backtest", json=BODY)
    assert res.status_code == 200
    assert res.json()["meta"]["data_source"] == "snapshot"


def test_both_sources_failing_returns_the_live_error():
    app = create_app(prices_loader=_fail_prices, rf_loader=_fail_rf, fallback=(_fail_prices, _fail_rf))
    res = TestClient(app).post("/api/backtest", json=BODY)
    assert res.status_code == 400
    assert "internet connection" in res.json()["detail"]


def test_reports_which_assets_were_eligible_at_the_last_rebalance(client):
    body = {**BODY, "tickers": ["TLT", "GLD", "SPY", "LATE"], "lookback": 252}
    data = client.post("/api/backtest", json=body).json()
    assert data["eligible"] == ["TLT", "GLD", "SPY"]
    assert data["weights"]["Equal weight"]["LATE"] == 0


def test_regime_endpoint_serves_study_files(tmp_path, monkeypatch):
    import api.main as m
    (tmp_path / "latest.json").write_text('{"as_of": "2026-11-17", "round2": {"chosen": {"signal": "hmm"}}}')
    (tmp_path / "backward.json").write_text('{"rule": {"signal": "hmm", "q": 0.9}}')
    monkeypatch.setattr(m, "REGIME_DIR", tmp_path)
    data = TestClient(m.create_app(prices_loader=fake_prices, rf_loader=fake_rf)).get("/api/regime").json()
    assert data["study"]["as_of"] == "2026-11-17"
    assert data["backward"]["rule"]["q"] == 0.9


def test_regime_endpoint_404_when_study_missing(tmp_path, monkeypatch):
    import api.main as m
    monkeypatch.setattr(m, "REGIME_DIR", tmp_path)
    res = TestClient(m.create_app(prices_loader=fake_prices, rf_loader=fake_rf)).get("/api/regime")
    assert res.status_code == 404
    assert "run_regime_study.py" in res.json()["detail"]
