# MAE Hillclimb and 2026 Shadow Test

Date: 2026-09-17

## Result

The 2026 shadow test found that `selected_team_strength` does not generalize well.
Its MAE is 26.639887 on 215 completed 2026 games. The simpler `scoring_shots`
model has an MAE of 25.693767 on the same games.

The new `conservative_selected_team` model uses this fixed formula:

```text
0.7 * selected_team_strength + 0.3 * scoring_shots
```

The 0.7 weight had the lowest 2015 to 2022 MAE on a fixed 0.05 grid. The model
reduces MAE in the development period, the 2023 to 2024 validation period, and
the 2026 shadow period.

| Period | Selected team | Conservative model | Gain |
|---|---:|---:|---:|
| 2015 to 2022 | 27.316967 | 27.211259 | 0.105708 |
| 2023 to 2024 | 25.829883 | 25.807390 | 0.022493 |
| 2025 | 25.396387 | 25.554065 | -0.157678 |
| 2026 | 26.639887 | 26.097102 | 0.542785 |

The 2025 result is worse. This is a known tradeoff. The conservative model gives
less weight to a player model that had a large 2026 error.

## Same-game comparison with Wheelo

Our predictions were complete and hashed before the Wheelo tip archive was
downloaded. All 215 completed games have one prediction from each model.

| Model | 2026 MAE |
|---|---:|
| Wheelo Ratings | 24.821442 |
| Scoring shots | 25.693767 |
| Conservative selected team | 26.097102 |
| Selected team strength | 26.639887 |

The scoring model trails Wheelo by 0.872325 points. The paired bootstrap 95%
interval is -0.329225 to 2.069087. One season does not show a stable difference.

## Published team model experiment

The Wheelo methodology describes adjusted scores, attack and defense ratings,
travel, five-year venue experience, venue performance, and recent player Rating
Points. A research model tested the first four items in 2,001 fixed settings.

The development winner had these results:

| Period | Scoring model | Published-structure model | Gain |
|---|---:|---:|---:|
| 2015 to 2022 | 27.587261 | 27.114599 | 0.472662 |
| 2023 to 2024 | 26.411850 | 26.444497 | -0.032647 |
| 2025 | 26.630732 | 26.831386 | -0.200654 |
| 2026 | 25.693767 | 25.212951 | 0.480816 |

The model missed the validation gate by 0.033 points, so it is not in production.
Travel produced most of the gain and most of the variation between seasons. The
next version must fit travel and venue terms in nested annual folds. It must use
a fixed rule before it reads the next season.

## Rejected player experiments

- Splitting Rating Points into performance rate and playing-time exposure made
  every development setting worse.
- A pooled player forecast improved individual Rating Points MAE by 0.083 on
  validation, but it made match MAE worse by 0.147.
- Adding the pooled forecast as a second player feature also failed validation.
- Team style, player role, and the combined style and role model failed the
  validation gate.

These tests show that better player-stat forecasts do not always give better
margin forecasts. The match model needs a stable team control and a small player
correction.

## Data and importer changes

- The current AFL Tables page can add a notes link to a round heading. The season
  parser now accepts this form.
- Modern AFL match pages have 23 players per team. The player importer now accepts
  either 22 or 23.
- The 2026 shadow input has 215 matches and 9,890 player rows from each player
  source.
- AFL Tables and Squiggle differ by one point for Essendon against Port Adelaide
  on 2026-08-23. The shadow test uses the AFL Tables score because its goals and
  behinds equal that score.

## Verification

```text
175 tests passed
```

The model rules use only information from completed earlier matches and the final
participants at kickoff. They do not use betting odds or competitor predictions.

Sources:

- https://www.wheeloratings.com/afl_methodology.html
- https://api.squiggle.com.au/?q=games&year=2026
- https://api.squiggle.com.au/?q=tips&year=2026&source=26
- https://afltables.com/afl/seas/2026.html
