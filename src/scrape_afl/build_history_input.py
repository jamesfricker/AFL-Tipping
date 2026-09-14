"""Add dated historical results to a research input without changing current rows."""

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

from src.mae_model.data import load_matches_csv
from src.mae_model.venues import get_venue_meta


DATA_DIR = Path(__file__).parent


def build_model_input(history_csv: Path, current_csv: Path, output_csv: Path) -> dict:
    if output_csv.resolve() in {history_csv.resolve(), current_csv.resolve()}:
        raise ValueError("Output must not overwrite an input file")
    zones = json.loads((DATA_DIR / "historical_venue_timezones.json").read_text())
    exceptions = json.loads(
        (DATA_DIR / "historical_result_availability.json").read_text()
    )
    with history_csv.open() as handle:
        raw = list(csv.DictReader(handle))
    with current_csv.open() as handle:
        reader = csv.DictReader(handle)
        original_fields = reader.fieldnames
        current = list(reader)
    if not raw or not current or not original_fields:
        raise ValueError("Both historical and current match inputs must contain rows")
    if max(int(row["year"]) for row in raw) >= min(int(row["year"]) for row in current):
        raise ValueError("Historical seasons must precede current seasons")
    historical_years = {int(row["year"]) for row in raw}
    required_exceptions = {
        match_id
        for match_id, item in exceptions.items()
        if item["year"] in historical_years
    }
    if required_exceptions - {row["match_id"] for row in raw}:
        raise ValueError("Missing historical result exception match IDs")
    zone_manifest = {}
    used_exceptions = {}
    for row in raw:
        if row["time_status"] != "source_local_time":
            raise ValueError(f"Unknown time for {row['match_id']}")
        meta = get_venue_meta(row["venue"])
        if meta:
            zone, location, source = (
                meta.timezone,
                meta.canonical_name,
                "src/mae_model/venues.py",
            )
        elif row["venue"] in zones:
            zone, location, source = zones[row["venue"]]
        else:
            raise ValueError(f"Unknown timezone for {row['venue']}")
        row["venue_timezone"] = zone
        row["timezone_source"] = source
        row["kickoff"] = (
            datetime.strptime(f"{row['date']} {row['time']}", "%d-%b-%Y %I:%M %p")
            .replace(tzinfo=ZoneInfo(zone))
            .isoformat()
        )
        row["result_available_at"] = ""
        row["result_availability_status"] = "assumed_next_venue_local_midnight"
        row["result_availability_note"] = ""
        row["result_availability_source"] = ""
        if row["match_id"] in exceptions:
            exception = exceptions[row["match_id"]]
            if exception["year"] != int(row["year"]):
                raise ValueError(
                    f"Historical result exception year differs for {row['match_id']}"
                )
            row["result_available_at"] = exception["result_available_at"]
            row["result_availability_status"] = exception["status"]
            row["result_availability_note"] = exception["note"]
            row["result_availability_source"] = exception["source"]
            used_exceptions[row["match_id"]] = exception
        zone_manifest[row["venue"]] = {
            "timezone": zone,
            "location": location,
            "source": source,
        }
    fields = list(dict.fromkeys(original_fields + list(raw[0])))
    combined = raw + current
    if len({row["match_id"] for row in combined}) != len(combined):
        raise ValueError("Duplicate match IDs")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_csv.parent) as staging:
        temporary_csv = Path(staging) / "matches.csv"
        with temporary_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(combined)
        loaded = load_matches_csv(str(temporary_csv))
        original_loaded = load_matches_csv(str(current_csv))
        current_ids = {match.match_id for match in original_loaded}
        if [
            match for match in loaded if match.match_id in current_ids
        ] != original_loaded:
            raise ValueError("Loaded current matches changed")
        with temporary_csv.open() as handle:
            copied_current = [
                {key: row[key] for key in original_fields}
                for row in csv.DictReader(handle)
                if row["match_id"] in current_ids
            ]
        if copied_current != current:
            raise ValueError("Original current CSV fields changed")
        temporary_csv.replace(output_csv)
    manifest = {
        "historical_matches": len(raw),
        "current_matches": len(current),
        "total_matches": len(loaded),
        "original_current_fields_unchanged": True,
        "loaded_current_matches_unchanged": True,
        "result_availability": "Assume next venue-local midnight except for listed exceptions. This is not a historical publication archive.",
        "kickoff_provenance": "The first time listed on each season page. The source does not attest these were known before the match.",
        "evaluation_policy": "Keep the original evaluation match IDs. A larger history changes the default minimum-training start year.",
        "sha256": hashlib.sha256(output_csv.read_bytes()).hexdigest(),
        "history_source_sha256": hashlib.sha256(history_csv.read_bytes()).hexdigest(),
        "current_source_sha256": hashlib.sha256(current_csv.read_bytes()).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "venue_map_sha256": hashlib.sha256(
            (DATA_DIR / "historical_venue_timezones.json").read_bytes()
        ).hexdigest(),
        "availability_map_sha256": hashlib.sha256(
            (DATA_DIR / "historical_result_availability.json").read_bytes()
        ).hexdigest(),
        "venue_timezones": zone_manifest,
        "result_availability_exceptions": used_exceptions,
    }
    output_csv.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-csv", type=Path, required=True)
    parser.add_argument("--current-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    result = build_model_input(args.history_csv, args.current_csv, args.output_csv)
    print(f"Wrote {result['total_matches']} matches. SHA-256 {result['sha256']}")


if __name__ == "__main__":
    main()
