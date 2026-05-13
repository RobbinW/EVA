import argparse
import csv
import json
from pathlib import Path

import cv2
from tqdm import tqdm


CSV_COLUMNS = [
    "video_path",
    "fps",
    "n_frames",
    "width",
    "height",
    "caption",
    "prompt_embed_path",
]


def get_video_info(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        "fps": fps,
        "n_frames": n_frames,
        "width": width,
        "height": height,
    }


def _first_nonempty_text(value):
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return ""


def get_caption(instruction_path: Path):
    if not instruction_path.exists():
        return ""

    try:
        with instruction_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        print(f"Error reading caption from {instruction_path}: {exc}")
        return ""

    for key in ("seen", "unseen", "instruction", "caption"):
        caption = _first_nonempty_text(data.get(key))
        if caption:
            return caption
    return ""


def discover_video_files(root_dir: Path):
    return sorted(root_dir.glob("*/*/video/*.mp4"))


def build_prompt_embed_path(video_path: Path, root_dir: Path) -> str:
    return str(video_path.relative_to(root_dir).with_suffix(".pt"))


def build_record(video_path: Path, root_dir: Path):
    base_dir = video_path.parent.parent
    instruction_path = base_dir / "instructions" / f"{video_path.stem}.json"

    video_meta = get_video_info(video_path)
    if not video_meta:
        print(f"Warning: Could not read video {video_path}")
        return None

    return {
        "video_path": str(video_path),
        "fps": video_meta["fps"],
        "n_frames": video_meta["n_frames"],
        "width": video_meta["width"],
        "height": video_meta["height"],
        "caption": get_caption(instruction_path),
        "prompt_embed_path": build_prompt_embed_path(video_path, root_dir),
    }


def collect_records(root_dir: Path):
    video_files = discover_video_files(root_dir)
    print(f"Found {len(video_files)} videos matching pattern <task>/<variant>/video/*.mp4")

    records = []
    for video_path in tqdm(video_files):
        record = build_record(video_path, root_dir)
        if record is not None:
            records.append(record)
    return records


def write_csv(records, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir",
        required=True,
        help="RoboTwin dataset root directory.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path. Defaults to <root-dir>/robotwin_videos.csv.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root_dir = Path(args.root_dir).expanduser().resolve()
    output_csv = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv is not None
        else root_dir / "robotwin_videos.csv"
    )

    records = collect_records(root_dir)
    if not records:
        print("No records found.")
        return

    write_csv(records, output_csv)
    print(f"Successfully wrote {len(records)} records to {output_csv}")


if __name__ == "__main__":
    main()
