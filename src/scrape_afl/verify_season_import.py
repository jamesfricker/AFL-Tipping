"""Verify saved season data and its 2012 overlap with the current match CSV."""

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

from src.scrape_afl.scrape_tables import MATCH_FIELDS
from src.scrape_afl.season_scores import parse_season_scores


def verify_import(output_dir: Path, current_csv: Path) -> dict:
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_years = list(range(manifest["first_year"], manifest["last_year"] + 1))
    if sorted(source["year"] for source in manifest["sources"]) != expected_years:
        raise ValueError("Manifest source years are missing or duplicated")
    parsed = []
    for source in manifest["sources"]:
        content = (output_dir / "cache" / f"{source['year']}.html").read_bytes()
        if hashlib.sha256(content).hexdigest() != source["sha256"]:
            raise ValueError(f"Cache hash mismatch for {source['year']}")
        rows, _ = parse_season_scores(content, source["year"])
        parsed.extend(rows)
    raw_csv = output_dir / "season_scores.csv"
    if hashlib.sha256(raw_csv.read_bytes()).hexdigest() != manifest["csv_sha256"]:
        raise ValueError("Imported CSV hash mismatch")
    with raw_csv.open() as handle:
        recorded = list(csv.DictReader(handle))
    if recorded != [{key: str(value) for key, value in row.items()} for row in parsed]:
        raise ValueError("Imported CSV does not match parsed source pages")
    overlap_page = output_dir / "cache" / "2012.html"
    if not overlap_page.exists():
        raise ValueError(
            "Save the 2012 overlap source page to cache/2012.html before verification"
        )
    overlap, _ = parse_season_scores(overlap_page.read_bytes(), 2012)
    with current_csv.open() as handle:
        current = {
            row["match_id"]: row
            for row in csv.DictReader(handle)
            if row["year"] == "2012"
        }
    if len(current) != len(overlap) or set(current) != {
        row["match_id"] for row in overlap
    }:
        raise ValueError("2012 overlap match IDs differ")
    for row in overlap:
        for field in MATCH_FIELDS:
            actual, expected = str(row[field]), current[row["match_id"]][field]
            if field == "date":
                actual = datetime.strptime(actual, "%d-%b-%Y").date()
                expected = datetime.strptime(expected, "%d-%b-%Y").date()
            if actual != expected:
                raise ValueError(f"2012 overlap differs for {row['match_id']} {field}")
    result = {
        "verified_source_pages": len(manifest["sources"]),
        "verified_imported_matches": len(parsed),
        "overlap_year": 2012,
        "overlap_matches": len(overlap),
        "compared_fields": MATCH_FIELDS,
        "date_comparison": "Parsed calendar date. Leading zeros do not affect equality.",
        "paired_id_sha256": hashlib.sha256(
            json.dumps(sorted(current)).encode()
        ).hexdigest(),
        "overlap_source_sha256": hashlib.sha256(overlap_page.read_bytes()).hexdigest(),
        "current_csv_sha256": hashlib.sha256(current_csv.read_bytes()).hexdigest(),
        "import_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "verification_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "parser_sha256": hashlib.sha256(
            Path(__file__).with_name("season_scores.py").read_bytes()
        ).hexdigest(),
    }
    (output_dir / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--current-csv", type=Path, required=True)
    args = parser.parse_args()
    result = verify_import(args.output_dir, args.current_csv)
    print(
        f"Verified {result['verified_imported_matches']} imported matches and {result['overlap_matches']} overlap matches"
    )


if __name__ == "__main__":
    main()
