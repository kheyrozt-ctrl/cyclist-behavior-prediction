from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_GIT_FILE_BYTES = 50 * 1024 * 1024
DATA_SUFFIXES = {".csv", ".tsv", ".txt", ".json", ".yaml", ".yml"}
FORBIDDEN_PATHS = (
    re.compile(r"(?:^|/)VideoCrop/fold_[1-5]_(?:train|val|test)_files\.txt$"),
    re.compile(r"(?:^|/)VideoCrop/gesture_crop_info.*\.csv$"),
    re.compile(r"(?:^|/)ManueverDataset/fold_[1-5]_(?:train|val|test)_data\.csv$"),
)
FORBIDDEN_VALUES = (
    re.compile(r"Participant_\d+_\d"),
    re.compile(r"\b20\d{6}_\d{6}(?:_\d{3})?\b"),
    re.compile(r"\bcam2rgbfixed\b"),
)


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [
        ROOT / value.decode("utf-8")
        for value in result.stdout.split(b"\0")
        if value
    ]


def main() -> int:
    failures: list[str] = []
    for path in tracked_files():
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if path.stat().st_size > MAX_GIT_FILE_BYTES:
            failures.append(f"{relative}: exceeds 50 MiB")
        if any(pattern.search(relative) for pattern in FORBIDDEN_PATHS):
            failures.append(f"{relative}: generated source-identifier manifest")
        if path.suffix.lower() not in DATA_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in FORBIDDEN_VALUES:
            if pattern.search(text):
                failures.append(f"{relative}: contains forbidden source identifier")
                break

    if failures:
        print("Public privacy check failed:", file=sys.stderr)
        for failure in sorted(set(failures)):
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("Public privacy check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
