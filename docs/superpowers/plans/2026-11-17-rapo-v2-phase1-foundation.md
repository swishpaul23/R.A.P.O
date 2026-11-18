# RAPO v2 Phase 1 (Foundation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace R.A.P.O's flat scripts with a tested package whose backtest is correct (simple-return aggregation, weight drift, costs, no look-ahead), so every strategy, including HRP, is compared honestly across full and stress periods.

**Architecture:** A `rapo/` package with one responsibility per module: data (cached prices and risk-free rate) → returns → estimators → strategies (all share one signature) → backtest engine → metrics → report/viz. The Streamlit app becomes a thin layer over `rapo.report`. Tests run offline: all network access goes through an injectable downloader.

**Tech Stack:** Python ≥3.10, NumPy, pandas, SciPy, scikit-learn (Ledoit-Wolf only), yfinance, pyarrow (Parquet cache), Matplotlib/Seaborn, Streamlit, pytest.

**Spec:** Notion → Coding / PROJECTS DOCS / R.A.P.O v2 / Design Doc (https://app.notion.com/p/41bd43a6f4fa8152a359d1312b196129), sections 6 "Phase 1" and 3 "Anti-goals". Phases 2–4 get their own plans after this one ships.

**Working mode:** Claude implements each task, then walks Stuart through it (what it does, why, and how to defend it in an interview) before starting the next.

## Global Constraints

- Long-only: every weight vector is ≥ 0 and sums to 1 (tolerance 1e-8).
- Annualization factor: 252 trading days.
- Backtest defaults: lookback 252 days, rebalance every 63 days, transaction cost 10 bps on turnover.
- No look-ahead: a decision at day `t` sees only returns strictly before `t`.
- Portfolio daily return = Σ wᵢ·rᵢ using **simple** returns; never sum weighted log returns.
- No global `dropna()` on prices: an asset is eligible at a rebalance only if it has a complete lookback window.
- Risk-free rate comes from data (^IRX), never a hardcoded constant.
- Max-Sharpe per-asset cap default 0.40 *(Claude's call; spec says "caps" without a value)*.
- Stress windows: "2020 COVID crash" = 2020-02-01 to 2020-06-30; "2022 rate shock" = 2022-01-01 to 2022-12-31 *(Claude's call)*.
- Tests never hit the network.

## Review Focus

1. **Bad ticker / empty download.** Expect a `ValueError` naming the ticker(s), not a KeyError deep in pandas. Test in Task 2.
2. **Rebalance with 0 or 1 eligible asset** (e.g. only new tickers). Expect 1 asset → 100% weight; 0 assets → hold cash (return 0, no cost). Test in Task 5.
3. **Data shorter than the lookback.** Expect a `ValueError` saying how many days are needed. Test in Task 5.
4. **Max-Sharpe when every expected return is below the risk-free rate** (the optimizer degenerates). Expect valid weights by falling back to min-variance. Test in Task 3.
5. **Near-duplicate assets** (SPY and VOO, correlation ≈ 1, singular covariance). Expect every strategy to still return valid weights. Tests in Tasks 3 and 4.

---

## File Structure

| File | Responsibility |
|---|---|
| `rapo/__init__.py` | Package marker, version |
| `rapo/returns.py` | Price → simple/log returns, preserving pre-inception NaN |
| `rapo/data.py` | Cached price download, risk-free series |
| `rapo/estimators.py` | Mean and covariance estimates (sample, Ledoit-Wolf) |
| `rapo/strategies.py` | Equal weight, inverse vol, min variance, max Sharpe |
| `rapo/hrp.py` | Hierarchical Risk Parity (ported from `portfolio_math.py`, fixed) |
| `rapo/backtest.py` | Walk-forward engine: eligibility, drift, costs |
| `rapo/metrics.py` | Performance metrics, stress-window slicing |
| `rapo/montecarlo.py` | Vectorized random-portfolio cloud |
| `rapo/report.py` | Run all strategies, build the comparison tables |
| `rapo/viz.py` | Figures (equity, drawdown, heatmap, dendrogram, frontier) |
| `app.py` | Streamlit UI over `rapo.report` / `rapo.viz` |
| `tests/test_*.py` | One test file per module |
| Deleted: `main.py`, `portfolio_math.py`, `data_loader.py`, `visualizer.py`, `test_setup.py` | Replaced by the package |

Strategy signature used everywhere:

```python
Strategy = Callable[[pd.DataFrame], pd.Series]
# input: daily simple returns, rows = days in lookback window, columns = eligible tickers, no NaN
# output: weights indexed by the same tickers, >= 0, sum to 1
```

---

### Task 1: Package scaffold and returns

**Files:**
- Create: `rapo/__init__.py`, `rapo/returns.py`, `tests/test_returns.py`, `pytest.ini`
- Modify: `requirements.txt` (rewrite as UTF-8, LF line endings)

**Interfaces:**
- Produces: `simple_returns(prices: pd.DataFrame) -> pd.DataFrame`, `log_returns(prices: pd.DataFrame) -> pd.DataFrame`

- [ ] **Step 1: Rewrite `requirements.txt`** as UTF-8 with: `numpy`, `pandas`, `scipy`, `scikit-learn`, `yfinance`, `pyarrow`, `matplotlib`, `seaborn`, `streamlit`, `pytest` (unpinned lower bounds matching the current pins where present). Add `pytest.ini` with `testpaths = tests`.

- [ ] **Step 2: Write the failing tests**

```python
def test_simple_returns_values():
    p = pd.DataFrame({"A": [100.0, 110.0, 99.0]}, index=pd.date_range("2024-01-01", periods=3))
    r = simple_returns(p)
    assert r["A"].iloc[0] == pytest.approx(0.10)
    assert r["A"].iloc[1] == pytest.approx(-0.10)
    assert len(r) == 2

def test_simple_returns_preserves_pre_inception_nan():
    p = pd.DataFrame({"OLD": [1.0, 1.1, 1.2, 1.3], "NEW": [np.nan, np.nan, 5.0, 5.5]},
                     index=pd.date_range("2024-01-01", periods=4))
    r = simple_returns(p)
    assert len(r) == 3                      # rows NOT dropped because NEW is missing
    assert r["NEW"].isna().sum() == 2
    assert r["NEW"].iloc[-1] == pytest.approx(0.10)

def test_log_returns_match_simple():
    p = pd.DataFrame({"A": [100.0, 110.0]}, index=pd.date_range("2024-01-01", periods=2))
    assert log_returns(p)["A"].iloc[0] == pytest.approx(np.log(1.1))
```

- [ ] **Step 3: Run** `python -m pytest tests/test_returns.py -v`. Expected: FAIL (module not found).

- [ ] **Step 4: Implement** both functions in `rapo/returns.py` using `pct_change(fill_method=None)` and dropping only the first row.

- [ ] **Step 5: Run** `python -m pytest tests/test_returns.py -v`. Expected: 3 passed.

- [ ] **Step 6: Commit** `feat: package scaffold, utf-8 requirements, returns module`

---

### Task 2: Data layer with cache and risk-free rate

**Files:**
- Create: `rapo/data.py`, `tests/test_data.py`

**Interfaces:**
- Consumes: none
- Produces:
  - `Downloader = Callable[[str, pd.Timestamp, pd.Timestamp], pd.Series]` (adjusted close for one ticker; empty Series if unknown)
  - `yf_downloader(ticker, start, end) -> pd.Series` (real implementation; `yf.download(..., auto_adjust=True)["Close"]`)
  - `fetch_prices(tickers: list[str], start: str, end: str, cache_dir: Path = Path(".cache/prices"), downloader: Downloader = yf_downloader) -> pd.DataFrame` (columns in input order, outer-joined on date, NaN before each ticker's inception)
  - `risk_free_daily(start: str, end: str, cache_dir=..., downloader=...) -> pd.Series` (from `^IRX`: `irx / 100 / 252`, forward-filled)

- [ ] **Step 1: Write the failing tests** using a fake downloader that counts calls:

```python
def test_fetch_prices_outer_joins_and_keeps_order(tmp_path): ...
    # fake: "A" has 5 days, "B" starts on day 3 → df.columns == ["A", "B"], df["B"].isna().sum() == 2

def test_fetch_prices_uses_cache_second_time(tmp_path): ...
    # call twice with the same range → fake downloader called once per ticker total

def test_fetch_prices_downloads_only_missing_tail(tmp_path): ...
    # first call to 2024-01-10, second call to 2024-01-20 → second download's start > 2024-01-10

def test_fetch_prices_bad_ticker_raises_with_name(tmp_path):
    with pytest.raises(ValueError, match="ZZZZ"):
        fetch_prices(["A", "ZZZZ"], "2024-01-01", "2024-02-01", tmp_path, fake)

def test_risk_free_daily_converts_percent(tmp_path): ...
    # fake ^IRX = 5.04 → every value == pytest.approx(0.0504 / 252)
```

- [ ] **Step 2: Run** `python -m pytest tests/test_data.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement** in `rapo/data.py`. One Parquet file per ticker (`{cache_dir}/{ticker}.parquet`, with `^` replaced by `_`). On a cache hit covering the range, slice it; otherwise download only `(last_cached_date + 1 day, end)` and append.

- [ ] **Step 4: Run** `python -m pytest tests/test_data.py -v`. Expected: 5 passed.

- [ ] **Step 5: Commit** `feat: cached price loader and data-driven risk-free rate`

---

### Task 3: Estimators and core strategies

**Files:**
- Create: `rapo/estimators.py`, `rapo/strategies.py`, `tests/test_strategies.py`

**Interfaces:**
- Consumes: Strategy signature (File Structure section)
- Produces:
  - `annual_mean(returns: pd.DataFrame) -> pd.Series` (mean × 252)
  - `ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame` (sklearn `LedoitWolf` × 252, labelled by ticker)
  - `equal_weight(returns) -> pd.Series`
  - `inverse_vol(returns) -> pd.Series`
  - `min_variance(returns) -> pd.Series` (SLSQP on LW covariance, bounds [0, 1])
  - `make_max_sharpe(rf_annual: float = 0.0, cap: float = 0.40) -> Strategy` (SLSQP on `annual_mean` + LW covariance, bounds [0, max(cap, 1/n)]; falls back to `min_variance` if every mean ≤ rf or the optimizer fails)

- [ ] **Step 1: Write the failing tests**

```python
ALL = [equal_weight, inverse_vol, min_variance, make_max_sharpe()]

@pytest.mark.parametrize("strat", ALL)
def test_weights_valid(strat, random_returns):          # fixture: 300 days × 4 assets, seed 0
    w = strat(random_returns)
    assert list(w.index) == list(random_returns.columns)
    assert (w >= -1e-10).all() and w.sum() == pytest.approx(1.0)

def test_inverse_vol_ratio():
    # asset vols 1% and 2% daily, uncorrelated → weights 2/3 and 1/3 (approx 1e-2)

def test_min_variance_prefers_low_vol(): ...   # low-vol asset gets > 0.5

def test_max_sharpe_respects_cap(random_returns):
    assert make_max_sharpe(cap=0.40)(random_returns).max() <= 0.40 + 1e-6

def test_max_sharpe_falls_back_when_all_below_rf(random_returns):
    w = make_max_sharpe(rf_annual=10.0)(random_returns)   # absurd rf → every mean below it
    pd.testing.assert_series_equal(w, min_variance(random_returns), atol=1e-6)

@pytest.mark.parametrize("strat", ALL)
def test_near_duplicate_assets(strat, random_returns):
    r = random_returns.assign(DUP=random_returns.iloc[:, 0] + 1e-9)
    w = strat(r)
    assert w.sum() == pytest.approx(1.0) and (w >= -1e-10).all()
```

- [ ] **Step 2: Run** `python -m pytest tests/test_strategies.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement** `rapo/estimators.py` and `rapo/strategies.py` to the signatures above. Clip tiny negatives to 0 and renormalize before returning.

- [ ] **Step 4: Run** `python -m pytest tests/test_strategies.py -v`. Expected: all pass.

- [ ] **Step 5: Commit** `feat: estimators and core allocation strategies`

---

### Task 4: HRP

**Files:**
- Create: `rapo/hrp.py`, `tests/test_hrp.py`

**Interfaces:**
- Consumes: Strategy signature
- Produces:
  - `hrp(returns: pd.DataFrame) -> pd.Series` (a Strategy)
  - `hrp_linkage(returns: pd.DataFrame) -> np.ndarray` (single-linkage matrix on distance `sqrt((1 - corr) / 2)`, for the dendrogram)

- [ ] **Step 1: Write the failing tests**

```python
def test_hrp_valid(random_returns): ...           # >= 0, sums to 1, index matches columns

def test_hrp_two_uncorrelated_is_inverse_variance():
    # vols 1% and 2%, uncorrelated, 5000 days, seed 0 → weights ≈ [0.8, 0.2] (abs 0.02)

def test_hrp_single_asset():
    assert hrp(one_col_returns).iloc[0] == 1.0

def test_hrp_near_duplicates(random_returns):
    r = random_returns.assign(DUP=random_returns.iloc[:, 0] + 1e-9)
    w = hrp(r); assert w.sum() == pytest.approx(1.0) and np.isfinite(w).all()
```

- [ ] **Step 2: Run** `python -m pytest tests/test_hrp.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement** by porting `get_ivp`, `get_cluster_var`, `get_quasi_diag` and `get_rec_bisection` from `portfolio_math.py` into `rapo/hrp.py`. Fixes: return a Series in input column order; handle n = 1 before clustering; clip the distance matrix at 0 before `squareform` (duplicates can produce tiny negatives inside the sqrt).

- [ ] **Step 4: Run** `python -m pytest tests/test_hrp.py -v`. Expected: 4 passed.

- [ ] **Step 5: Commit** `feat: HRP as a standard strategy`

---

### Task 5: Walk-forward backtest engine

**Files:**
- Create: `rapo/backtest.py`, `tests/test_backtest.py`

**Interfaces:**
- Consumes: `simple_returns`, Strategy signature
- Produces:
  - `@dataclass BacktestResult: returns: pd.Series` (daily net of costs, starting at the first rebalance day), `weights: pd.DataFrame` (target weights per rebalance date, all tickers as columns, 0 if ineligible), `turnover: pd.Series` (per rebalance date, Σ|target − drifted|), `costs: pd.Series` (per rebalance date)
  - `eligible_assets(returns: pd.DataFrame, t: int, lookback: int) -> list[str]` (columns with no NaN in rows `[t - lookback, t)` and a non-NaN return at row `t`)
  - `run_backtest(returns: pd.DataFrame, strategy: Strategy, lookback: int = 252, rebalance_every: int = 63, cost_bps: float = 10.0) -> BacktestResult`

**Semantics (decided here, not left to the implementer):**
- Rebalance rows `t = lookback, lookback + rebalance_every, …`. The strategy gets `returns.iloc[t - lookback : t][eligible]`.
- Cost on day `t` = `cost_bps / 1e4 × turnover`, subtracted from day `t`'s portfolio return.
- Day return = Σ wᵢ·rᵢ over held assets. A NaN return for a held asset counts as 0 that day.
- After each day, drift: `wᵢ ← wᵢ(1 + rᵢ) / (1 + r_p_gross)`.
- 0 eligible → all-cash: return 0, weights 0, no cost.
- `len(returns) <= lookback` → `ValueError` whose message includes the required number of days.

- [ ] **Step 1: Write the failing tests**

```python
def test_two_asset_hand_example():
    # 2 assets, lookback=2, rebalance_every=100 (one rebalance), cost 0, equal weight.
    # Day returns after lookback: A = [+10%, 0%], B = [0%, +10%]
    # Day 1: 0.5*0.10 = 0.05.  Drift → wA = 0.55/1.05, wB = 0.5/1.05
    # Day 2: (0.5/1.05)*0.10 = 0.047619...
    assert res.returns.tolist() == pytest.approx([0.05, 0.5 / 1.05 * 0.10])

def test_costs_charged_on_turnover():
    # Same data, cost_bps=10: first rebalance from cash, turnover 1.0 → day-1 return 0.05 - 0.001

def test_no_lookahead():
    seen = []
    def spy(window): seen.append(window.index.max()); return equal_weight(window)
    res = run_backtest(rets, spy, lookback=20, rebalance_every=5)
    for max_seen, rebal_date in zip(seen, res.weights.index):
        assert max_seen < rebal_date

def test_new_asset_enters_only_after_full_window(): ...
    # NEW starts at row 30, lookback 20 → weight 0 at rebalances before row 50, > 0 after

def test_single_eligible_asset_gets_full_weight(): ...
def test_zero_eligible_holds_cash(): ...                    # returns 0, cost 0
def test_too_short_raises():
    with pytest.raises(ValueError, match="253"):
        run_backtest(rets.iloc[:100], equal_weight, lookback=252)
```

- [ ] **Step 2: Run** `python -m pytest tests/test_backtest.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement** `rapo/backtest.py` to the semantics above. A simple day loop is fine (correctness first; this loop is the engine, not the "vectorized" claim).

- [ ] **Step 4: Run** `python -m pytest tests/test_backtest.py -v`. Expected: 7 passed.

- [ ] **Step 5: Commit** `feat: correct walk-forward backtest with drift, costs, eligibility`

---

### Task 6: Metrics and stress windows

**Files:**
- Create: `rapo/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Consumes: `BacktestResult.returns`, `risk_free_daily` output
- Produces:
  - `STRESS_WINDOWS: dict[str, tuple[str, str]]` (values from Global Constraints)
  - `cagr(r) -> float`, `ann_vol(r) -> float`, `sharpe(r, rf) -> float`, `sortino(r, rf) -> float`, `max_drawdown(r) -> float` (negative number), `calmar(r) -> float`, `var_hist(r, level=0.95) -> float` (5th percentile of daily returns, negative)
  - `summary(r: pd.Series, rf: pd.Series, turnover: pd.Series | None = None) -> pd.Series` (all of the above plus `annual_turnover`)
  - `window(r: pd.Series, name: str) -> pd.Series` (slice by `STRESS_WINDOWS[name]`)

- [ ] **Step 1: Write the failing tests**

```python
def test_cagr_doubling_over_252_days():
    r = pd.Series([2 ** (1 / 252) - 1] * 252)
    assert cagr(r) == pytest.approx(1.0)

def test_max_drawdown_known_path():
    r = pd.Series([0.10, -0.50, 0.20])        # 1.1 → 0.55 → 0.66
    assert max_drawdown(r) == pytest.approx(-0.50)

def test_sharpe_zero_when_excess_mean_is_zero():
    idx = pd.date_range("2024-01-01", periods=252, freq="B")
    rf = pd.Series(0.0001, index=idx)
    noise = np.random.default_rng(0).normal(0, 0.01, 252)
    r = rf + (noise - noise.mean())
    assert sharpe(r, rf) == pytest.approx(0.0, abs=1e-12)
def test_var_hist_is_5th_percentile(): ...
def test_sortino_ignores_upside(): ...           # adding a big up-day raises sortino, not downside dev
def test_window_slices_dates(): ...
```

- [ ] **Step 2: Run** `python -m pytest tests/test_metrics.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement** in `rapo/metrics.py`. Align `rf` to `r`'s index with forward-fill. Sharpe = mean(excess) × 252 / (std(excess) × √252).

- [ ] **Step 4: Run** `python -m pytest tests/test_metrics.py -v`. Expected: all pass.

- [ ] **Step 5: Commit** `feat: performance metrics and stress windows`

---

### Task 7: Vectorized Monte Carlo

**Files:**
- Create: `rapo/montecarlo.py`, `tests/test_montecarlo.py`

**Interfaces:**
- Produces: `random_portfolios(mu: pd.Series, cov: pd.DataFrame, n: int = 10_000, rf: float = 0.0, seed: int = 0) -> pd.DataFrame` (columns `ret`, `vol`, `sharpe`, plus one weight column per ticker)

- [ ] **Step 1: Write the failing tests**

```python
def test_matches_loop_computation():
    df = random_portfolios(mu, cov, n=50, seed=1)
    W = df[mu.index].to_numpy()
    for i in range(50):
        assert df["vol"].iloc[i] == pytest.approx(np.sqrt(W[i] @ cov.to_numpy() @ W[i]))
        assert df["ret"].iloc[i] == pytest.approx(W[i] @ mu.to_numpy())

def test_weights_valid(): ...   # each row >= 0, sums to 1
def test_is_fast():             # n=100_000 with 10 assets finishes < 1s
```

- [ ] **Step 2: Run** `python -m pytest tests/test_montecarlo.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement** with `rng.dirichlet(np.ones(k), size=n)` for weights, `W @ mu` for returns, and `np.einsum("ij,jk,ik->i", W, cov, W)` for variances. No Python loop.

- [ ] **Step 4: Run** `python -m pytest tests/test_montecarlo.py -v`. Expected: 3 passed.

- [ ] **Step 5: Commit** `feat: vectorized Monte Carlo frontier`

---

### Task 8: Comparison report and figures

**Files:**
- Create: `rapo/report.py`, `rapo/viz.py`, `tests/test_report.py`, `tests/test_viz.py`

**Interfaces:**
- Consumes: everything above
- Produces:
  - `DEFAULT_STRATEGIES: dict[str, Strategy]` = `{"Equal weight", "Inverse vol", "Min variance", "Max Sharpe", "HRP"}`
  - `@dataclass Report: results: dict[str, BacktestResult], table: pd.DataFrame` (rows = strategies + benchmark ticker, cols = `summary` fields), `stress_tables: dict[str, pd.DataFrame]`
  - `build_report(prices: pd.DataFrame, rf: pd.Series, benchmark: str, strategies=DEFAULT_STRATEGIES, **backtest_kwargs) -> Report` (the benchmark is buy-and-hold over the same dates; it is excluded from the strategies' investable columns)
  - `viz.equity_curves(report) -> Figure`, `viz.drawdowns(report) -> Figure`, `viz.correlation_heatmap(returns) -> Figure` (seaborn `heatmap`), `viz.dendrogram(returns) -> Figure`, `viz.frontier(mc: pd.DataFrame, highlight: dict[str, tuple[float, float]]) -> Figure`

- [ ] **Step 1: Write the failing tests**

```python
def test_build_report_runs_all_strategies(synthetic_prices, flat_rf):
    rep = build_report(synthetic_prices, flat_rf, benchmark="BENCH", lookback=60, rebalance_every=20)
    assert set(rep.table.index) == set(DEFAULT_STRATEGIES) | {"BENCH"}
    assert "max_drawdown" in rep.table.columns
    assert set(rep.stress_tables) == set(STRESS_WINDOWS)   # synthetic prices span 2019–2023

def test_benchmark_not_in_strategy_universe(synthetic_prices, flat_rf):
    rep = build_report(...)
    assert "BENCH" not in rep.results["HRP"].weights.columns

@pytest.mark.parametrize("fn", [...all viz functions...])
def test_figures_render(fn, ...):
    assert isinstance(fn(...), matplotlib.figure.Figure)   # Agg backend
```

- [ ] **Step 2: Run** `python -m pytest tests/test_report.py tests/test_viz.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement** `rapo/report.py` and `rapo/viz.py`. The risk-free rate passed to max Sharpe is the mean of `rf` × 252 over the data range.

- [ ] **Step 4: Run** `python -m pytest -v` (full suite). Expected: all pass.

- [ ] **Step 5: Commit** `feat: strategy comparison report and figures`

---

### Task 9: Streamlit app and removal of old scripts

**Files:**
- Modify: `app.py` (rewrite)
- Delete: `main.py`, `portfolio_math.py`, `data_loader.py`, `visualizer.py`, `test_setup.py`
- Modify: `.gitignore` (add `.cache/`)

**Interfaces:**
- Consumes: `fetch_prices`, `risk_free_daily`, `build_report`, `random_portfolios`, `viz.*`

- [ ] **Step 1: Rewrite `app.py`.** The sidebar keeps the existing inputs (tickers, benchmark, dates) and adds lookback, rebalance and cost. One "Research" tab with: comparison table (full period), one table per stress window, equity curves, drawdowns, correlation heatmap, dendrogram, and a frontier with each strategy's final-window position highlighted. Errors from `fetch_prices`/`run_backtest` are shown with `st.error(str(e))`. Wrap downloads in `st.cache_data`.

- [ ] **Step 2: Delete the old scripts** and confirm nothing imports them: `grep -rn "portfolio_math\|data_loader\|visualizer" --include=*.py .` → no output.

- [ ] **Step 3: Run** `python -m pytest -v`. Expected: all pass.

- [ ] **Step 4: Manual check.** Run `streamlit run app.py` with defaults (`SPY, TLT, GLD, NVDA, XLP`, benchmark SPY, 2015-01-01 to today). Every section renders. HRP appears in the comparison table. Then enter ticker `ZZZZ` and confirm a readable error appears.

- [ ] **Step 5: Commit** `feat: rewire Streamlit app onto rapo package; remove legacy scripts`

---

## After Phase 1

- Walk Stuart through the comparison table: which strategies held up in 2020 and 2022, and why.
- Update the R.A.P.O write-up so every line is true, using real numbers.
- Write the Phase 2 plan (regime detector) from the design doc plus what Phase 1 showed about the data.
