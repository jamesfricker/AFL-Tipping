# MAE 24.0 Feasibility Check (2025)

## Question
Is 2025 MAE ~= 24 achievable with current data and modeling approach?

## Baseline numbers (from `.context/reports_baseline_check/mae_summary.csv`)
- market_only 2025 MAE: **26.1944**
- market_line_blend 2025 MAE: **26.0380**
- market_residual_corrector 2025 MAE: **26.0729**

Gap to target 24.0 from best baseline = **2.038 points/game**.
Across 216 games, this is ~**440 absolute-error points** to remove.

## Data quality / available market fields
Checked `src/outputs/afl_betting_history.xlsx`:
- Current loader uses only: `Home Line Close`, `Home Odds`, `Away Odds`.
- Workbook also contains richer fields: open/min/max lines, totals open/close/min/max, line/total odds and other close/open columns.
- 2025 market close coverage appears full in joined backtest rows (216 games).

## Experiments run
All experiments were run as strict train(<2025) -> test(2025), unless explicitly labeled oracle/in-sample.

### 1) Market-only anchor variants using richer market columns
- market_close MAE: **26.1944**
- market_open MAE: **26.2269**
- avg(open,close) MAE: **26.1204**

No meaningful improvement toward 24.

### 2) Direct market-feature models (train<2025, test2025)
Features included close/open/min/max line, odds (open/close), totals (open/close/min/max), and movement-derived terms.
- Ridge MAE: **26.1479**
- Huber MAE: **26.1881**
- HistGradientBoosting (MAE loss) MAE: **26.4115**

Best was still around 26.15.

### 3) Constrained residual correction from market_close
Huber residual model + cap/scale search on correction:
- best MAE: **26.1369**

Still far from 24.

### 4) Gated-correction stress test using model disagreement + market context
Using baseline predictions + market movement/odds/totals as gating/correction features:
- Best residual-corrector setup: **26.0551**
- Trainable hard gate (choose anchor vs corrected): **26.0401**
- Trainable soft gate: **26.0375** (essentially unchanged vs anchor 26.0380)

So learned gating did not unlock large gains under year-locked training.

### 5) Upper/lower-bound diagnostics
- Oracle choose-better-per-game between anchor and one learned corrector: **25.3440**
- Oracle linear blend trained on 2025 itself (not valid OOS):
  - Ridge in-sample MAE: **25.8173**
  - Quantile in-sample MAE: **25.4006**

Interpretation: even optimistic in-sample linear methods with current fields do not approach 24.

### 6) Within-2025 CV check (signal sanity)
5-fold CV on 2025 only with market-rich features:
- Ridge CV MAE: **27.4291**
- Quantile CV MAE: **28.0759**
- HGB CV MAE: **27.8856**

This indicates weak stable within-season predictive signal beyond the close line itself.

## Feasibility conclusion
With the **current effective information set** (current match data + available market workbook fields) and strict year-locked setup, reaching **2025 MAE ~24.0** appears **not realistically achievable**.

Evidence:
- Best robust out-of-sample results remain clustered around **26.0-26.2**.
- Large required improvement (2+ points/game) is not recovered by richer market engineering or learned gating under temporal discipline.

## What would likely be required to approach 24
1. Stronger market anchor inputs than currently exploited/available in model ingestion:
   - true multi-book consensus close at timestamped final line,
   - exchange/price depth features,
   - explicit line movement time series (not just open-close delta).
2. High-value external pre-game signals not currently represented strongly enough:
   - late outs/injury confirmations,
   - high-quality weather + venue effects tied to kickoff timing.
3. Distributional modeling of market (median-optimal inference) instead of point-line correction, likely with richer market microstructure data.
