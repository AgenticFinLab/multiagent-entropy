import csv
import json
from pathlib import Path
from typing import Dict, List, Any


def ensure_directory_exists(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], output_path: str) -> None:
    output_file = Path(output_path)
    ensure_directory_exists(output_file)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(
    rows: List[Dict[str, Any]], output_path: str, fieldnames: List[str]
) -> None:
    output_file = Path(output_path)
    ensure_directory_exists(output_file)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_output_directory(base_path: Path, dataset: str, subdirectory: str = "") -> Path:
    output_dir = base_path / "evaluation" / "results" / dataset
    if subdirectory:
        output_dir = output_dir / subdirectory
    return output_dir


def format_float(value: float, precision: int = 4) -> float:
    return round(value, precision)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator != 0 else default
