# RAPO v2 Phase 2 (Regime Detector) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Predict the probability that the next month is a market-stress period, use it to tilt HRP / equal weight toward T-bills and gold, and prove, with walk-forward tests, bootstrap intervals and honest baselines, whether that beats plain HRP and 60/40 in both 2020-type and 2022-type crashes.

**Architecture:** A new `rapo/regime/` package: features → labels → walk-forward probabilities (VIX rule, HMM, logistic, gradient boosting) → evaluation → an overlay `Strategy` that plugs into the existing backtest. Training is slow (hundreds of refits), so `scripts/run_regime_study.py` runs offline on the committed price snapshot and writes `data/regime/latest.json`. The API serves that file; the UI adds a stress reading and a cycle chart.

**Tech Stack:** Python, pandas, scikit-learn (LogisticRegression, HistGradientBoostingClassifier, isotonic calibration), hmmlearn (offline only), FastAPI, React.

**Spec:** Notion → Coding / PROJECTS DOCS / R.A.P.O v2 / Design Doc, sections 6 (Phase 2) and 9 (Phase 1 findings).

## Global Constraints

- No look-ahead anywhere: a value dated `d` uses only data dated ≤ `d`. A rebalance at row `t` uses probabilities dated ≤ the window's last day (`t − 1`). Every module that touches time has a test that changes the future and checks the past doesn't move.
- Stress label (fixed before any model is evaluated): day `d` is stress = 1 if SPY's lowest close in the next 21 trading days is ≤ 8% below its close on `d`. The last 21 days are NaN.
- Walk-forward training: expanding window, refit every 21 days, 21-day gap between the last training label and the prediction date (labels look 21 days ahead), first prediction after 750 training days.
- Probabilities that drive allocation are isotonic-calibrated inside the training fold only.
- Overlay: tilt = 0.40 × clip((p̂ − 0.20) / 0.80, 0, 1), where p̂ is a 5-day EMA of the probability. Defensive sleeve = 50% SHY, 50% GLD. Overlay backtests rebalance every 21 days; plain HRP / equal weight are also reported at 21 days so the comparison is like-for-like.
- Benchmarks: SPY buy-and-hold and 60/40 (SPY/AGG, rebalanced every 63 days).
- A result counts as a win only if the 95% block-bootstrap interval (21-day blocks, 1,000 resamples, seed 0) of the difference excludes zero.
- `hmmlearn` goes in `requirements-dev.txt` only; the deployed API never trains.

## Review Focus

1. **Missing feature inputs** (^VIX or ^TNX absent from the snapshot). Expect a `ValueError` naming the missing tickers, not NaN features. Test in Task 2.
2. **Too little history before the first prediction.** Expect probabilities to stay NaN until 750 training days, and the overlay to treat NaN as "no tilt". Tests in Tasks 4 and 7.
3. **A training fold with only one class** (no stress days yet). Expect the model to return the fold's base rate instead of crashing. Test in Task 4.
4. **Defensive assets not yet eligible** (BIL starts 2007). Expect the overlay to scale the base weights back to 100% with no crash. Test in Task 7.
5. **Stale `latest.json`.** Expect the API to report the study's `as_of` date and the UI to show it. Test in Task 10.

---

### Task 1: Snapshot tickers and the 60/40 benchmark

**Files:** Modify `scripts/make_snapshot.py`, `rapo/strategies.py`, `rapo/report.py`; Test `tests/test_strategies.py`, `tests/test_report.py`

**Interfaces:**
- Produces: `fixed_weights(target: dict[str, float]) -> Strategy` (renormalizes over the eligible subset); `build_report(..., extra_benchmarks: dict[str, Strategy] | None = None)`. Extra benchmarks run through the same backtest and appear as table rows with their own names (e.g. "60/40").
- `make_snapshot.py` adds `^VIX`, `^TNX`, `HYG`, `SHY`, `BIL`.

- [ ] **Step 1:** Tests: `fixed_weights({"A": .6, "B": .4})` on a frame with A, B, C returns A .6, B .4, C 0; with only A eligible returns A 1.0. `build_report(..., extra_benchmarks={"60/40": fixed_weights(...)})` table contains a "60/40" row.
- [ ] **Step 2:** Run `python -m pytest tests/test_strategies.py tests/test_report.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement, and add the five tickers to `TICKERS` in the snapshot script.
- [ ] **Step 4:** Run the same tests. Expected: PASS. Run the full suite. Expected: all pass.
- [ ] **Step 5:** Commit `feat: fixed-weight strategy, 60/40 benchmark, snapshot tickers for regime features`. **Stuart re-runs `python scripts/make_snapshot.py` and commits before Task 9.**

### Task 2: Features

**Files:** Create `rapo/regime/__init__.py`, `rapo/regime/features.py`; Test `tests/regime/test_features.py`

**Interfaces:**
- Produces: `REQUIRED = ["SPY", "TLT", "IEF", "HYG", "^VIX", "^TNX"]`; `build_features(prices: pd.DataFrame) -> pd.DataFrame` with columns `vol_21, vol_63, vol_ratio, vix, vix_chg_21, corr_eq_bond_63, credit_63, rate_mom_63, curve, mom_200`, indexed by date.

| Column | Definition (all on data ≤ d) |
|---|---|
| `vol_21`, `vol_63` | std of SPY daily returns × √252 |
| `vol_ratio` | `vol_21 / vol_63` |
| `vix`, `vix_chg_21` | ^VIX level; change over 21 days |
| `corr_eq_bond_63` | 63-day rolling correlation of SPY and TLT returns |
| `credit_63` | 63-day return of HYG minus 63-day return of IEF |
| `rate_mom_63` | ^TNX level change over 63 days (percentage points) |
| `curve` | ^TNX − ^IRX if ^IRX is present, else NaN column |
| `mom_200` | SPY / 200-day moving average − 1 |

- [ ] **Step 1:** Tests: `test_no_lookahead` (multiply all prices after row 400 by 2; rows ≤ 400 of the features are identical); `test_missing_inputs_raise` (drop ^VIX → `ValueError` matching "\\^VIX"); `test_columns`; `test_warmup_only_nan` (no NaN after row 200).
- [ ] **Step 2:** Run `python -m pytest tests/regime/test_features.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement with rolling windows only (no centered windows, no `shift(-k)`).
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat(regime): feature builder`.

### Task 3: Stress labels

**Files:** Create `rapo/regime/labels.py`; Test `tests/regime/test_labels.py`

**Interfaces:** Produces `stress_label(spy: pd.Series, horizon: int = 21, threshold: float = -0.08) -> pd.Series` (float 1.0 / 0.0, NaN for the last `horizon` rows).

- [ ] **Step 1:** Tests: a hand path `[100, 100, 91, 100, …]` with horizon 2 labels row 0 as 1 (91/100 − 1 = −9%) and row 2 as 0; the last 2 rows are NaN; a path that falls exactly 8% is labeled 1.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement with a forward rolling min (`spy[::-1].rolling(horizon).min()[::-1].shift(-1)`) divided by `spy`.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit `feat(regime): forward-drawdown stress labels`.

### Task 4: Walk-forward probabilities (VIX rule, logistic, boosting)

**Files:** Create `rapo/regime/models.py`, `rapo/regime/walkforward.py`; Test `tests/regime/test_walkforward.py`

**Interfaces:**
- Produces:
  - `ModelFactory = Callable[[], ClassifierMixin]`; `logistic() -> Pipeline` (StandardScaler + LogisticRegression C=0.5, class_weight="balanced"); `boosting() -> HistGradientBoostingClassifier` (max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0).
  - `vix_rule(features: pd.DataFrame, threshold: float = 30) -> pd.Series` (1.0 if vix > threshold else 0.0).
  - `walk_forward_proba(features, labels, factory, min_train=750, refit_every=21, gap=21, calibrate=True) -> pd.Series`. For each refit date `d`, train on rows with date ≤ d − gap rows and non-NaN label; predict rows `d … d + refit_every − 1`. NaN before `min_train`. Calibration: `CalibratedClassifierCV(method="isotonic", cv=TimeSeriesSplit(3))`, falling back to uncalibrated if a split has one class. A training set with one class returns its base rate for the block.

- [ ] **Step 1:** Tests: `test_training_never_sees_the_future` (a factory wrapper records `max(train_index)` for each fit; each is < prediction start − gap); `test_nan_before_min_train`; `test_single_class_fold_returns_base_rate`; `test_probabilities_in_unit_interval`; `test_vix_rule`.
- [ ] **Step 2:** Run `python -m pytest tests/regime/test_walkforward.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit `feat(regime): walk-forward calibrated stress probabilities`.

### Task 5: HMM baseline

**Files:** Create `rapo/regime/hmm.py`; Modify `requirements-dev.txt` (+ `hmmlearn>=0.3`); Test `tests/regime/test_hmm.py`

**Interfaces:** Produces `hmm_stress_proba(spy_returns: pd.Series, min_train=750, refit_every=63, seed=0) -> pd.Series`. GaussianHMM, 2 states, full covariance, fit on `[return, 21-day vol]`; stress state = the state with the higher mean volatility (fixes label switching); value at `d` = filtered (forward-algorithm) probability of the stress state using data ≤ d.

- [ ] **Step 1:** Tests: on a synthetic series alternating calm (σ = 0.5%) and stressed (σ = 3%) blocks of 250 days, mean probability in stressed blocks > 0.7 and in calm blocks < 0.3 (after `min_train`); `test_no_lookahead` (same as Task 2's pattern).
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. Use `model.predict_proba` on the history ending at `d` and take the last row, so smoothing never uses data after `d`.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit `feat(regime): HMM baseline`.

### Task 6: Forecast evaluation and bootstrap

**Files:** Create `rapo/regime/evaluate.py`; Test `tests/regime/test_evaluate.py`

**Interfaces:** Produces `forecast_scores(proba, labels) -> dict` (keys `auc`, `brier`, `base_rate`, `n`; aligned on non-NaN rows); `calibration_table(proba, labels, bins=5) -> pd.DataFrame` (columns `mean_pred`, `observed`, `count`); `block_bootstrap_ci(a: pd.Series, b: pd.Series, metric: Callable[[pd.Series], float], block=21, n=1000, seed=0) -> tuple[float, float, float]` (point estimate of metric(a) − metric(b), 2.5% and 97.5% bounds; resamples the same blocks for a and b).

- [ ] **Step 1:** Tests: perfect predictions give AUC 1 and Brier 0; constant 0.5 gives Brier 0.25; the bootstrap of a series against itself returns (0, 0, 0); a series with +0.1% daily edge over its twin gives an interval above 0 for mean return; the result is reproducible with the same seed.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement (`sklearn.metrics.roc_auc_score`, `brier_score_loss`).
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit `feat(regime): forecast scores and block bootstrap`.

### Task 7: Regime overlay strategy

**Files:** Create `rapo/regime/overlay.py`; Test `tests/regime/test_overlay.py`

**Interfaces:**
- Consumes: Strategy signature, `pd.Series` probabilities.
- Produces: `tilt_from_proba(p: pd.Series, cap=0.40, floor=0.20, span=5) -> pd.Series`; `make_regime_overlay(base: Strategy, proba: pd.Series, defensive: dict[str, float] = {"SHY": .5, "GLD": .5}, cap=0.40, floor=0.20, span=5) -> Strategy`. Weights = (1 − tilt) × base(window without defensive-only tickers) + tilt × defensive (renormalized over eligible defensive tickers), where tilt = `tilt_from_proba(...)` looked up with `.asof(window.index[-1])`; NaN → 0.

- [ ] **Step 1:** Tests: p ≡ 0 → identical to base; p ≡ 1 → 0.6 × base + 0.4 × defensive; p = 0.6 → tilt 0.2; NaN probabilities → base; SHY ineligible → all defensive weight on GLD; GLD and SHY both ineligible → base renormalized to 1; weights always ≥ 0 and sum to 1.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run, then run the whole suite. Expected: PASS.
- [ ] **Step 5:** Commit `feat(regime): probability-driven defensive overlay`.

### Task 8: Overlay metrics

**Files:** Modify `rapo/metrics.py`; Test `tests/test_metrics.py`

**Interfaces:** Produces `rebound_capture(r: pd.Series, bench: pd.Series, trough: str, days: int = 63) -> float` (the strategy's return over the `days` after `trough` divided by the benchmark's); `whipsaw_count(tilt: pd.Series, weeks: int = 4, threshold: float = 0.10) -> int` (number of tilt increases of ≥ threshold reversed within `weeks`).

- [ ] **Step 1:** Tests with hand-built series for both.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit `feat: rebound capture and whipsaw metrics`.

### Task 9: Regime study script and results

**Files:** Create `scripts/run_regime_study.py`, `rapo/regime/study.py`, `docs/results/phase2.md` (generated), `data/regime/latest.json` (generated); Test `tests/regime/test_study.py`

**Interfaces:** Produces `run_study(prices: pd.DataFrame, universe: list[str], start: str) -> dict` with keys `as_of`, `forecast` (scores per model: vix_rule, hmm, logistic, boosting), `calibration` (per model), `backtests` (metrics table: SPY, 60/40, EW@21, HRP@21, EW+overlay, HRP+overlay, per chosen model), `stress_windows` (same tables for 2008-09, 2020 and 2022), `bootstrap` (overlay − base CIs for Sharpe and max drawdown), `series` (dates, chosen model's probability, tilt, HRP+overlay equity), `current` (latest probability and tilt). The chosen model is the one with the lowest out-of-sample Brier score; ties go to the simpler model.

- [ ] **Step 1:** Test `run_study` end-to-end on synthetic prices containing all required tickers: every key is present, the JSON is serializable with no NaN, and the series lengths match.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. The script loads `data/snapshot/`, runs the study for universe `SPY, QQQ, EFA, EEM, TLT, IEF, GLD, VNQ` from 2007-01-01, and writes both outputs. `phase2.md` reports every number, including the ones that don't favour the overlay.
- [ ] **Step 4:** Run the test (expected: PASS), then `python scripts/run_regime_study.py` on the real snapshot. Expected: both files are written.
- [ ] **Step 5:** Commit `feat(regime): study runner and Phase 2 results`.

### Task 10: API and UI

**Files:** Modify `api/main.py`, `web/src/App.tsx`; Create `web/src/components/RegimePanel.tsx`, `web/src/components/CycleChart.tsx`; Tests `tests/test_api.py`, `web/src/components/RegimePanel.test.tsx`

**Interfaces:**
- `GET /api/regime` returns `data/regime/latest.json` as-is (404 with "Run scripts/run_regime_study.py" if missing).
- UI: new **Regime** section below the KPIs (still pattern 9). It shows a stress-probability tile with an (i) explainer, the current tilt, the model name and `as_of` date; a cycle chart (SPY price line, background shaded by probability, markers where the tilt rose or fell by ≥ 10 points); and the overlay comparison table for full period, 2008-09, 2020 and 2022 with bootstrap intervals.

- [ ] **Step 1:** Tests: the API returns the file's JSON and `as_of`; it returns 404 with that message when the file is missing. `RegimePanel` renders the probability as a percentage, the tilt, the model name and the `as_of` date, and shows "Study data is from <date>" when `as_of` is older than 45 days.
- [ ] **Step 2:** Run both suites. Expected: FAIL.
- [ ] **Step 3:** Implement. Run `ux-audit` at 375px and 1280px before finishing.
- [ ] **Step 4:** Run both suites. Expected: PASS.
- [ ] **Step 5:** Commit `feat: regime panel and cycle chart`.

## After Phase 2

Phase 4 write-up: README results from `docs/results/phase2.md`, resume bullets using the bootstrap intervals, and the design-doc anti-goals checked line by line.
