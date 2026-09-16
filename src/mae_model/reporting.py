import csv
import hashlib
import json
import subprocess
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .sequential_margin import MODEL_CONFIG


def write_metadata(
    output_dir,
    input_paths,
    rows,
    matches,
    config,
    quotes,
    *,
    player_config=None,
    player_diagnostics=None,
):
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    base_rows = [row for row in rows if row.model_name == "market_scoring_blend"]
    weights = {
        (row.year, row.weight_cutoff.isoformat()): {
            "year": row.year,
            "market_weight": row.market_weight,
            "training_games": row.weight_training_games,
            "cutoff": row.weight_cutoff.isoformat(),
        }
        for row in base_rows
    }
    data = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_revision": revision,
        "git_dirty": bool(status),
        "configuration": {**MODEL_CONFIG, **config},
        "inputs": [
            {
                "path": str(Path(path).resolve()),
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            }
            for path in input_paths
        ],
        "source_hashes": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "time_assumptions": {
            "legacy_kickoff": "CSV date and time interpreted in the venue local timezone",
            "legacy_result": "Available at next local midnight; an assumption, not a verified publication time",
            "explicit_result": "result_available_at with an explicit timezone offset",
            "weight_cutoff": "1 January 00:00 Australia/Sydney of target season, or the earlier request cutoff",
            "market": "Untimed closing benchmark"
            if config.get("closing_line_benchmark")
            else "Latest observed_at at or before deadline and kickoff",
            "historical_training_deadline": "Kickoff minus lead_hours for each earlier match",
            "live_history": "Ignore result rows whose kickoff is at or after as_of; gate other results by availability",
        },
        "market_input_rows": len(quotes),
        "market_input_diagnostics": dict(
            Counter(quote.validation_note for quote in quotes if quote.validation_note)
        ),
        "market_input_missing_margins": sum(
            quote.predicted_margin is None for quote in quotes
        ),
        "history_matches": len(matches),
        "history_result_timing": dict(
            Counter(match.timing_assumption for match in matches)
        ),
        "history_shot_estimates": sum(
            not (match.home_scoring_shots or match.home_goals + match.home_behinds)
            or not (match.away_scoring_shots or match.away_goals + match.away_behinds)
            for match in matches
        ),
        "prediction_matches": len(base_rows),
        "market_status_counts": dict(Counter(row.market_status for row in base_rows)),
        "common_market_matches": sum(
            row.predicted_margin is not None
            for row in rows
            if row.model_name == "market_only"
        ),
        "fallback_count": sum(row.used_fallback for row in base_rows),
        "season_weights": sorted(
            weights.values(), key=lambda row: (row["year"], row["cutoff"])
        ),
        "tip_rule": "Predicted and actual margins must have the same sign. A draw is correct only for a zero predicted margin.",
        "evaluation_note": "Historical results describe this dataset. Previously inspected seasons are not an untouched test.",
    }
    if player_config is not None:
        data["player_model"] = {
            "configuration": asdict(player_config),
            "status_counts": dict(Counter(row.status for row in player_diagnostics)),
            "fallback_count": sum(
                row.status != "adjusted" for row in player_diagnostics
            ),
            "historical_lineups": "Final player identity assumed available at kickoff only",
            "live_lineups": "Latest complete 22-or-23-player snapshot observed by the deadline",
            "player_statistics": "The player result event waits for the match result and all player rows for that match",
            "rating": "Outcome logistic with a 400-point scale; exposure-weighted update; reliability shrinkage at forecast",
            "form": "Fixed box-score score per at least 50 percent game time; recent EMA minus career EMA",
            "career_form_rate": 0.05,
            "performance_weights": {
                "kicks": 3,
                "handballs": 2,
                "marks": 3,
                "goals": 6,
                "behinds": 1,
                "hit_outs": 1,
                "tackles": 4,
                "clearances": 3,
                "contested_possessions": 0.5,
                "goal_assists": 3,
                "clangers": -3,
            },
        }
        fields = [
            "match_id",
            "cutoff",
            "status",
            "correction",
            "home_missing_leader",
            "away_missing_leader",
            "home_missing_leader_gap",
            "away_missing_leader_gap",
            "rating_change_difference",
            "form_change_difference",
        ]
        with (Path(output_dir) / "player_diagnostics.csv").open(
            "w", newline=""
        ) as target:
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader()
            for row in player_diagnostics:
                diagnostic = {
                    "match_id": row.match_id,
                    "cutoff": row.cutoff.isoformat(),
                    "status": row.status,
                    "correction": row.correction,
                }
                if row.lineup is not None:
                    diagnostic.update(
                        {
                            "home_missing_leader": row.lineup.home.missing_leader,
                            "away_missing_leader": row.lineup.away.missing_leader,
                            "home_missing_leader_gap": row.lineup.home.missing_leader_gap,
                            "away_missing_leader_gap": row.lineup.away.missing_leader_gap,
                            "rating_change_difference": row.lineup.rating_change_difference,
                            "form_change_difference": row.lineup.form_change_difference,
                        }
                    )
                writer.writerow(diagnostic)
    (Path(output_dir) / "metadata.json").write_text(json.dumps(data, indent=2) + "\n")
