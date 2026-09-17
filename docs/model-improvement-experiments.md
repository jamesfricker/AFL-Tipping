# Market-free model improvement experiments

## Fixed evaluation

The primary measure is market-free margin MAE. Lower values are better. The development period is 2015 to 2022. The validation period is 2023 and 2024. The 2025 season is a descriptive target because earlier work inspected it.

The starting `team_only` MAE is 27.729524 in development and 26.268117 in validation. Its 2025 MAE is 26.348055.

## Accepted hybrid player model

The official Rating Points correction and the outcome and Fantasy correction contain different information. Their correction correlation is 0.065 from 2023 to 2025.

The hybrid calculation is:

```text
team_only + outcome correction + official correction
```

Each component keeps its existing four-point limit. The hybrid does not add a final limit. Both components use the same `team_only` control. The model keeps the AFL Tables and Wheelo player IDs separate.

| Evaluation set | Official player MAE | Hybrid player MAE | Improvement |
| --- | ---: | ---: | ---: |
| 2015 to 2022 development | 27.667784 | 27.596697 | 0.071087 |
| 2023 to 2024 validation | 26.236765 | 26.107115 | 0.129650 |
| 2025 descriptive target | 26.338423 | 26.122201 | 0.216222 |

The hybrid improves 9 of 11 seasons against `team_only`. It is 0.016743 points worse in 2017 and 0.029617 points worse in 2021. Overall MAE changes from 27.317777 to 27.170661.

The 2025 improvement against the official-only model is 0.216222 points. A paired bootstrap with 100,000 samples and seed `20260916` gave a 95% interval from 0.0050 to 0.4272 points.

## Accepted selected-team strength model

The lineup-change model ignores the full strength of a stable selected team. The selected-team model estimates each named player's next Rating Points. It uses the preceding 20 games and a 10-game league prior. It then fits the team forecast, the hybrid lineup change, and the difference between the two selected-team totals.

The annual fit uses only earlier seasons. The configuration maximizes the smaller gain across development and validation. This rule does not use the 2025 result.

| Evaluation set | Hybrid player MAE | Selected-team MAE | Improvement |
| --- | ---: | ---: | ---: |
| 2015 to 2022 development | 27.596697 | 27.316967 | 0.279730 |
| 2023 to 2024 validation | 26.107115 | 25.829883 | 0.277232 |
| 2025 descriptive target | 26.122201 | 25.396387 | 0.725814 |
| 2015 to 2025 | 27.170661 | 26.848736 | 0.321925 |

The 2025 result is 0.448521 points lower than Wheelo's matched MAE of 25.844907. The paired interval for that difference includes zero. The model does not have a reliable lead from one season.

The full-lineup design also tested annual weight updates during a season, fixed substitute weights, role-group totals, star and depth totals, and coaches-vote totals. None improved both development and validation against the selected configuration.

## Rejected tests

| Test | Result |
| --- | --- |
| Search 500 current team configurations | No setting improved both development and validation. |
| Search 10,000 direct margin ratings | The best validation MAE was 26.330460. |
| Search 1,000 opponent-adjusted team process models | The best validation MAE was 27.643192. |
| Blend the process model with the hybrid | The development-selected blend increased validation MAE to 26.474257. |
| Add static travel and venue corrections | Validation MAE increased from 26.2707 to 26.4509. |
| Add simple recent form and rest corrections | Validation MAE increased. |
| Use the first six matches to correct the rest of the season | The correlation was 0.084 and later-season MAE increased to 28.05. |

The Wheelo process source has complete inside-50, clearance, and contested-possession values for all 5,758 team rows from 2012 to 2025. Twelve old team rows have no shots-at-goal value. The tested process model did not earn a place in the product code.

## Limits

Historical player rows identify the final team that played. The model assumes that this team was known at kickoff. Wheelo does not give Rating Points publication times. The model assumes that a rating was available with the repository match result.
