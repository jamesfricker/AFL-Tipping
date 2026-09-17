import argparse
import csv
import sys
from collections import defaultdict
from statistics import fmean


def load_errors(path: str, model_name: str) -> dict[tuple[int, str], float]:
    errors = {}
    with open(path, newline="", encoding="utf-8-sig") as source:
        for line, row in enumerate(csv.DictReader(source), 2):
            if row["model_name"] != model_name:
                continue
            if not row["abs_error"]:
                raise ValueError(f"{path}:{line}: missing absolute error")
            key = (int(row["year"]), row["match_id"])
            if key in errors:
                raise ValueError(f"{path}:{line}: duplicate prediction {key}")
            errors[key] = float(row["abs_error"])
    if not errors:
        raise ValueError(f"No rows found for model {model_name!r}")
    return errors


def summarize(control: dict, candidate: dict) -> list[dict]:
    if control.keys() != candidate.keys():
        missing = len(control.keys() - candidate.keys())
        extra = len(candidate.keys() - control.keys())
        raise ValueError(
            f"Models do not cover the same matches: missing={missing}, extra={extra}"
        )
    by_year = defaultdict(list)
    for key in sorted(control):
        by_year[key[0]].append((control[key], candidate[key]))
    rows = []
    for year, values in sorted(by_year.items()):
        control_mae = fmean(value[0] for value in values)
        candidate_mae = fmean(value[1] for value in values)
        rows.append(
            {
                "year": str(year),
                "matches": len(values),
                "control_mae": control_mae,
                "candidate_mae": candidate_mae,
                "improvement": control_mae - candidate_mae,
            }
        )
    control_mae = fmean(control.values())
    candidate_mae = fmean(candidate.values())
    rows.append(
        {
            "year": "ALL",
            "matches": len(control),
            "control_mae": control_mae,
            "candidate_mae": candidate_mae,
            "improvement": control_mae - candidate_mae,
        }
    )
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two models on the same prediction rows."
    )
    parser.add_argument("predictions_csv")
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument("--min-non-worse-seasons", type=int, default=0)
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args(argv)

    try:
        control = load_errors(args.predictions_csv, args.control)
        candidate = load_errors(args.predictions_csv, args.candidate)
        rows = summarize(control, candidate)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print("year,matches,control_mae,candidate_mae,improvement")
    for row in rows:
        print(
            f"{row['year']},{row['matches']},{row['control_mae']:.6f},"
            f"{row['candidate_mae']:.6f},{row['improvement']:.6f}"
        )
    season_rows = rows[:-1]
    non_worse = sum(row["improvement"] >= 0 for row in season_rows)
    passed = (
        rows[-1]["improvement"] >= args.min_improvement
        and non_worse >= args.min_non_worse_seasons
    )
    print(
        f"gate={'PASS' if passed else 'FAIL'},"
        f"non_worse_seasons={non_worse}/{len(season_rows)}"
    )
    return 0 if passed or not args.gate else 1


if __name__ == "__main__":
    sys.exit(main())
