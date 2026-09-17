"""Create the frozen same-game 2026 comparison with Wheelo."""

import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.mae_model.data import canonical_team_name


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def percentile(values, probability):
    return values[int(probability * len(values))]


def main():
    games = json.loads((DATA / "squiggle_2026_games.json").read_text())["games"]
    tips = json.loads((DATA / "wheelo_2026_tips.json").read_text())["tips"]
    completed = {row["id"]: row for row in games if row.get("complete") == 100}
    wheelo = {row["gameid"]: row for row in tips if row["gameid"] in completed}
    if len(completed) != 215 or len(wheelo) != 215:
        raise ValueError("Expected 215 completed games and 215 Wheelo tips")

    predictions = defaultdict(dict)
    prediction_path = RESULTS / "run" / "walk_forward_predictions.csv"
    with prediction_path.open(newline="") as source:
        for row in csv.DictReader(source):
            if row["year"] != "2026":
                continue
            key = (
                row["kickoff"][:10],
                canonical_team_name(row["home_team"]),
                canonical_team_name(row["away_team"]),
            )
            predictions[key][row["model_name"]] = row

    output = []
    for game_id, game in sorted(completed.items()):
        key = (
            game["localtime"][:10],
            canonical_team_name(game["hteam"]),
            canonical_team_name(game["ateam"]),
        )
        model = predictions[key]
        candidate = model["preseason_structural_challenger"]
        scoring = model["scoring_shots"]
        repository_actual = float(candidate["actual_margin"])
        actual = float(game["hscore"]) - float(game["ascore"])
        candidate_margin = float(candidate["predicted_margin"])
        scoring_margin = float(scoring["predicted_margin"])
        wheelo_margin = float(wheelo[game_id]["hmargin"])
        output.append(
            {
                "squiggle_game_id": game_id,
                "match_id": candidate["match_id"],
                "date": key[0],
                "round": game["round"],
                "home_team": key[1],
                "away_team": key[2],
                "actual_margin": actual,
                "repository_actual_margin": repository_actual,
                "challenger_margin": candidate_margin,
                "challenger_abs_error": abs(actual - candidate_margin),
                "wheelo_margin": wheelo_margin,
                "wheelo_abs_error": abs(actual - wheelo_margin),
                "scoring_shots_margin": scoring_margin,
                "scoring_shots_abs_error": abs(actual - scoring_margin),
            }
        )

    RESULTS.mkdir(exist_ok=True)
    comparison_path = RESULTS / "comparison_2026.csv"
    with comparison_path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)

    paired = [
        row["challenger_abs_error"] - row["wheelo_abs_error"] for row in output
    ]
    rng = random.Random(20260917)
    bootstrap = sorted(
        fmean(rng.choice(paired) for _ in paired) for _ in range(100_000)
    )
    summary = {
        "games": len(output),
        "mae": {
            "preseason_structural_challenger": fmean(
                row["challenger_abs_error"] for row in output
            ),
            "wheelo_ratings": fmean(row["wheelo_abs_error"] for row in output),
            "scoring_shots": fmean(
                row["scoring_shots_abs_error"] for row in output
            ),
        },
        "challenger_minus_wheelo_mae": fmean(paired),
        "challenger_minus_wheelo_bootstrap_95_interval": [
            percentile(bootstrap, 0.025),
            percentile(bootstrap, 0.975),
        ],
        "lower_mae": "preseason_structural_challenger"
        if fmean(paired) < 0
        else "wheelo_ratings",
        "candidate_selection": (
            "Model settings use 2015-2022 development results. The 2023-2024 "
            "seasons are validation. The 2026 season was inspected during the "
            "wider research program before this final replay."
        ),
        "score_source": (
            "Squiggle completed-game scores. The repository differs by one "
            "point for Essendon against Port Adelaide on 2026-08-23."
        ),
        "prediction_csv_sha256": hashlib.sha256(
            prediction_path.read_bytes()
        ).hexdigest(),
        "comparison_csv_sha256": hashlib.sha256(
            comparison_path.read_bytes()
        ).hexdigest(),
        "wheelo_tips_sha256": hashlib.sha256(
            (DATA / "wheelo_2026_tips.json").read_bytes()
        ).hexdigest(),
    }
    (RESULTS / "comparison_2026.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
