from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import random
import statistics
from typing import Dict, Any


def gen_student_record(rng: random.Random) -> Dict[str, Any]:
    hours_studied_per_week = max(0.0, min(50.0, rng.gauss(15.0, 7.0)))
    attendance_rate = max(50.0, min(100.0, rng.gauss(88.0, 8.0)))
    previous_exam_score = max(0.0, min(100.0, rng.gauss(62.0, 18.0)))
    assignments_completed_ratio = max(0.0, min(1.0, rng.gauss(0.8, 0.15)))
    sleep_hours_per_night = max(4.0, min(10.0, rng.gauss(7.2, 1.0)))
    extracurricular_hours = max(0.0, min(15.0, rng.gauss(4.0, 2.5)))
    part_time_job_hours = max(0.0, min(30.0, rng.gauss(6.0, 5.0)))
    internet_access_quality = int(max(1, min(5, round(rng.gauss(4.0, 1.0)))))
    parent_education_level = int(max(0, min(3, round(rng.gauss(1.5, 1.0)))))
    tutoring_hours = max(0.0, min(8.0, rng.gauss(1.0, 1.5)))

    score = 0.0
    score += 0.35 * previous_exam_score
    score += 0.25 * (hours_studied_per_week * 2.0)  # 0..100-ish influence
    score += 0.15 * attendance_rate
    score += 20.0 * assignments_completed_ratio
    score += 0.6 * tutoring_hours
    score -= 0.9 * part_time_job_hours
    score += -2.0 * abs(sleep_hours_per_night - 7.5)
    score += 0.6 * extracurricular_hours - 0.04 * (extracurricular_hours ** 2)
    score += 2.0 * internet_access_quality
    score += 3.0 * parent_education_level
    score += rng.gauss(0.0, 8.0)

    final_score = max(0.0, min(100.0, score))
    label = 1 if final_score >= 60.0 else 0

    return {
        "hours_studied_per_week": round(hours_studied_per_week, 2),
        "attendance_rate": round(attendance_rate, 2),
        "previous_exam_score": round(previous_exam_score, 2),
        "assignments_completed_ratio": round(assignments_completed_ratio, 3),
        "sleep_hours_per_night": round(sleep_hours_per_night, 2),
        "extracurricular_hours": round(extracurricular_hours, 2),
        "part_time_job_hours": round(part_time_job_hours, 2),
        "internet_access_quality": internet_access_quality,
        "parent_education_level": parent_education_level,
        "tutoring_hours": round(tutoring_hours, 2),
        "final_score": round(final_score, 2),
        "label": label,
        "label_text": "pass" if label == 1 else "fail",
    }


def generate_dataset(n_rows: int, seed: int) -> list[Dict[str, Any]]:
    rng = random.Random(seed)
    return [gen_student_record(rng) for _ in range(n_rows)]


def write_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="") as f:
            pass
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def summarize(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"rows": len(rows)}
    if not rows:
        return out

    if "final_score" in rows[0]:
        scores = [r["final_score"] for r in rows]
        labels = [r["label"] for r in rows]
        passed = sum(1 for l in labels if l == 1)
        failed = len(labels) - passed
        out.update({
            "pass_count": passed,
            "fail_count": failed,
            "pass_rate": round(100.0 * passed / len(rows), 2),
            "score_mean": round(statistics.mean(scores), 2),
            "score_stdev": round(statistics.pstdev(scores), 2) if len(scores) > 1 else None,
            "score_min": round(min(scores), 2),
            "score_max": round(max(scores), 2),
        })
    else:
        marks = [r["marks"] for r in rows]
        failed = [r["failed"] for r in rows]
        fail_count = sum(int(f) for f in failed)
        pass_count = len(rows) - fail_count
        out.update({
            "pass_count": pass_count,
            "fail_count": fail_count,
            "pass_rate": round(100.0 * pass_count / len(rows), 2),
            "score_mean": round(statistics.mean(marks), 2),
            "score_stdev": round(statistics.pstdev(marks), 2) if len(marks) > 1 else None,
            "score_min": round(min(marks), 2),
            "score_max": round(max(marks), 2),
        })

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a labeled student dataset for logistic regression")
    parser.add_argument("--rows", "-n", type=int, default=3000,
                        help="Number of rows to generate (default: 3000; clamped to [2000,4000] unless --allow-out-of-range)")
    parser.add_argument("--out", type=str, default=str(Path("data") / "student_dataset.csv"),
                        help="Output CSV path (default: data/student_dataset.csv)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--allow-out-of-range", action="store_true",
                        help="Allow any --rows value without clamping (expert mode)")
    parser.add_argument("--schema", choices=["simple", "rich"], default="simple",
                        help="Output schema: 'simple' -> marks,failed | 'rich' -> many features (default: simple)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    n_rows = int(args.rows)
    if not args.allow_out_of_range:
        if n_rows < 2000:
            print(f"[info] Clamping rows from {n_rows} up to 2000 to meet the small dataset requirement.")
            n_rows = 2000
        if n_rows > 4000:
            print(f"[info] Clamping rows from {n_rows} down to 4000 to meet the small dataset requirement.")
            n_rows = 4000

    out_path = Path(args.out)
    base_rows = generate_dataset(n_rows, args.seed)

    if args.schema == "simple":
        simple_rows = []
        for r in base_rows:
            marks = int(round(float(r["final_score"])))
            failed = 1 if marks < 60 else 0
            simple_rows.append({"marks": marks, "failed": failed})
        rows_to_write = simple_rows
    else:
        rows_to_write = base_rows

    write_csv(out_path, rows_to_write)

    stats = summarize(rows_to_write)
    print(f"Wrote {stats['rows']} rows to {out_path}")
    print(f"Pass: {stats['pass_count']} | Fail: {stats['fail_count']} | Pass rate: {stats['pass_rate']}%")
    print(f"Score mean: {stats['score_mean']} ± {stats['score_stdev']} | min: {stats['score_min']} | max: {stats['score_max']}")


if __name__ == "__main__":
    main()
