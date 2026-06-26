from __future__ import annotations

import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


# Function to balance gesture data for one branch.
def balance_gesture_data(grouped_data, target_count, class_order, fold_number):
    # Go through each gesture type and limit its occurrences to target_count.
    rng = random.Random(20260620 + int(fold_number))
    balanced_data: list = []
    for gesture_type in class_order:
        items = list(grouped_data.get(gesture_type, []))
        if len(items) > int(target_count):
            # Randomly select target_count samples from the data.
            items = rng.sample(items, int(target_count))
        balanced_data.extend(items)
    rng.shuffle(balanced_data)
    return balanced_data


# 处理骨架文件，生成手势片段。
def process_skeleton_file(filename, start_ts, end_ts, gesture_type, skeleton_data_folder, output_frames=12, slide_interval=0.5, sample_info=None):
    segment_length = 1.0
    sample_info = sample_info or {}

    # 读取对应的骨骼数据文件。
    def read_skeleton(path: Path) -> tuple[np.ndarray, np.ndarray]:
        skeleton_data = pd.read_csv(path)
        if skeleton_data.shape[1] < 67:
            raise ValueError(f"Expected timestamp + 66 coordinate columns in {path}, got {skeleton_data.shape}")
        times = pd.to_numeric(skeleton_data.iloc[:, 0], errors="coerce").to_numpy(dtype=np.float32)
        coords = (
            skeleton_data.iloc[:, 1:67]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        good = np.isfinite(times)
        times, coords = times[good], coords[good]
        order = np.argsort(times)
        return times[order], coords[order]

    # 滑动窗口采样。
    def window_ranges(start: float, end: float) -> list[tuple[float, float]]:
        if end - start < segment_length - 1e-9:
            return []
        if end - start <= segment_length + 1e-9:
            return [(float(start), float(end))]
        starts: list[float] = []
        current = float(start)
        while current + segment_length <= end + 1e-9:
            starts.append(current)
            current += float(slide_interval)
        tail = float(end - segment_length)
        if not starts or abs(starts[-1] - tail) > 1e-6:
            starts.append(tail)
        return [(start, min(start + segment_length, end)) for start in starts]

    # 将窗口重采样到当前分支需要的固定帧数。
    def resample(times: np.ndarray, coords: np.ndarray, start: float, end: float) -> np.ndarray | None:
        if times.size == 0 or coords.size == 0:
            return None
        start, end = max(float(start), float(times[0])), min(float(end), float(times[-1]))
        if end <= start:
            return None
        mask = (times >= start) & (times <= end)
        seg_times, seg_coords = times[mask], coords[mask]
        if seg_times.size == 0:
            nearest = int(np.argmin(np.abs(times - ((start + end) / 2.0))))
            seg_times = np.asarray([times[nearest]], dtype=np.float32)
            seg_coords = coords[nearest : nearest + 1]
        if seg_times.size == 1:
            return np.repeat(seg_coords, int(output_frames), axis=0).astype(np.float32)

        target_times = np.linspace(start, end, int(output_frames), dtype=np.float32)
        sampled = np.empty((int(output_frames), coords.shape[1]), dtype=np.float32)
        for column in range(coords.shape[1]):
            sampled[:, column] = np.interp(target_times, seg_times, seg_coords[:, column])
        return sampled.astype(np.float32)

    skeleton_path = Path(skeleton_data_folder) / str(filename)
    times, coords = read_skeleton(skeleton_path)
    samples = []
    output_label = "neutral" if str(gesture_type).startswith("neutral_") else str(gesture_type)
    for start, end in window_ranges(float(start_ts), float(end_ts)):
        segment = resample(times, coords, start, end)
        if segment is None:
            continue
        samples.append(
            {
                "x": segment,
                "label": output_label,
                "clip_id": str(sample_info.get("clip_id", "")),
                "participant_id": str(sample_info.get("participant_id", "")),
                "skeleton_csv": str(sample_info.get("skeleton_csv", filename)),
                "start_sec": float(start),
                "end_sec": float(end),
            }
        )
    return samples


# 获取 neutral 时间段，排除动作区间和阻断区间。
def get_neutral_intervals(start_ts_list, end_ts_list, total_duration=24.0, margin=1.0, blocked_intervals=None, start_time=0.0):
    intervals = [
        (max(float(start_time), float(start) - margin), min(float(total_duration), float(end) + margin))
        for start, end in zip(start_ts_list, end_ts_list)
        if float(end) > float(start)
    ]
    intervals.extend(
        (max(float(start_time), float(start)), min(float(total_duration), float(end)))
        for start, end in (blocked_intervals or [])
        if float(end) > float(start)
    )
    intervals = sorted(intervals)
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    neutral_intervals = []
    current_start = float(start_time)
    for start, end in merged:
        if start > current_start:
            neutral_intervals.append((current_start, start))
        current_start = max(current_start, end)
    if current_start < float(total_duration):
        neutral_intervals.append((current_start, float(total_duration)))
    return [(start, end) for start, end in neutral_intervals if end > start]


# 处理文件并生成手势数据。
def process_files(file_list, gesture_info, skeleton_data_folder, fold_number, datasettype, gesture_definition):
    # 清洗标注表中的文本字段。
    def clean_text(value: object) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        return "" if text.lower() in {"nan", "none"} else text

    # 解析时间、帧号等数值字段。
    def parse_float(value: object) -> float | None:
        text = clean_text(value)
        if not text:
            return None
        try:
            out = float(text)
        except ValueError:
            return None
        return out if np.isfinite(out) else None

    def parse_int(value: object) -> int | None:
        number = parse_float(value)
        return int(number) if number is not None else None

    # 统一场景文件名，便于和 skeleton 文件对应。
    def normalize_clip_id(value: object) -> str:
        stem = Path(clean_text(value)).stem
        if stem.endswith("_skeleton"):
            stem = stem[: -len("_skeleton")]
        if stem and not stem.endswith("_cam2rgbfixed"):
            stem = f"{stem}_cam2rgbfixed"
        return stem

    def is_excluded(value: object) -> bool:
        return clean_text(value).lower() in {"exclude", "disable", "disabled"}

    # 从文件名中读取参与人编号。
    def participant_from_name(value: object) -> str:
        match = re.match(r"^Participant_0*(\d+)_", Path(clean_text(value)).stem)
        return match.group(1) if match else ""

    # 根据动作标签判断所属分支。
    def branch_for_gesture(gesture_type: str, reason_text: str) -> str:
        if gesture_type in {"Left_Look", "Right_Look"}:
            return "head"
        if gesture_type in {"Upper_Limb_Left_Rotation", "Upper_Limb_Right_Rotation"}:
            return "upper"
        if gesture_type.lower() == "pedaling":
            return "leg"
        match = re.search(r"\bbranch=(head|upper|upper_limb|leg)\b", reason_text, flags=re.IGNORECASE)
        if match:
            branch_name = match.group(1).lower()
            return "upper" if branch_name == "upper_limb" else branch_name
        return ""

    def canonical_label(branch_name: str, gesture_type: str) -> str:
        return "Pedaling" if branch_name == "leg" and gesture_type.lower() == "pedaling" else gesture_type

    # 读取当前标注行的动作时间段。
    def row_interval(row: pd.Series, fps: float | None) -> tuple[float, float] | None:
        start_frame = parse_int(row.get("Start Frame"))
        end_frame = parse_int(row.get("End Frame"))
        if fps and fps > 0 and start_frame is not None and end_frame is not None:
            start, end = start_frame / fps, (end_frame + 1) / fps
        else:
            start, end = parse_float(row.get("Start Timestamp")), parse_float(row.get("End Timestamp"))
        if start is None or end is None or end <= start:
            return None
        return float(start), float(end)

    # 合并重叠或相邻的阻断区间。
    def merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
        valid = sorted((float(start), float(end)) for start, end in intervals if end > start)
        merged: list[tuple[float, float]] = []
        for start, end in valid:
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return merged

    # 从动作区间中切掉不能进入训练样本的区间。
    def subtract_intervals(
        start_sec: float,
        end_sec: float,
        blocked: Sequence[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        pieces: list[tuple[float, float]] = []
        cursor = float(start_sec)
        for start, end in merge_intervals(
            (max(start_sec, start), min(end_sec, end))
            for start, end in blocked
            if min(end_sec, end) > max(start_sec, start)
        ):
            if start > cursor:
                pieces.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < end_sec:
            pieces.append((cursor, float(end_sec)))
        return [(start, end) for start, end in pieces if end > start]

    # 计算当前滑窗步长下能生成多少个窗口。
    def window_count(intervals: Sequence[dict], stride_sec: float, window_sec: float = 1.0) -> int:
        total = 0
        for item in intervals:
            duration = float(item["end_sec"]) - float(item["start_sec"])
            if duration < window_sec - 1e-9:
                continue
            if duration <= window_sec + 1e-9:
                total += 1
                continue
            count = int(np.floor((duration - window_sec) / stride_sec + 1e-9)) + 1
            tail = float(item["end_sec"]) - window_sec
            last = float(item["start_sec"]) + (count - 1) * stride_sec
            total += count + int(abs(last - tail) > 1e-6)
        return total

    def lower_median_cap(counts: Sequence[int]) -> int:
        nonzero = sorted(int(count) for count in counts if int(count) > 0)
        return nonzero[(len(nonzero) - 1) // 2] if nonzero else 0

    # 为每类样本选择可以达到平衡目标的滑窗步长。
    def choose_slide_interval(intervals: Sequence[dict], target: int) -> float:
        stride_candidates_sec = [0.25, 1.0 / 3.0, 0.5, 0.75, 1.0]
        if target <= 0:
            return 1.0
        counts = {stride: window_count(intervals, stride) for stride in stride_candidates_sec}
        for stride in sorted(stride_candidates_sec, reverse=True):
            if counts.get(stride, 0) >= target:
                return float(stride)
        return float(min(stride_candidates_sec))

    # 读取每个 fold/subset 的 exact-balanced target。
    def formal_exact_target(root: Path, fold: int, branch_name: str, subset: str) -> int | None:
        path = root / "Gesture2Manuever_Prediction" / "VideoCrop" / "stage1_exact_targets.txt"
        if not path.exists():
            return None
        targets = pd.read_csv(path, sep=r"\s+")
        match = targets[
            (targets["fold"].astype(int) == int(fold))
            & (targets["branch"].astype(str) == str(branch_name))
            & (targets["subset"].astype(str) == str(subset))
        ]
        return None if match.empty else int(match.iloc[0]["target_per_class"])

    root = Path(__file__).resolve().parents[2]
    branch = str(gesture_definition).strip().lower()
    if branch == "upper_limb":
        branch = "upper"
    if branch == "head":
        class_order = ["Left_Look", "Right_Look", "neutral_head"]
        positive_labels = {"Left_Look", "Right_Look"}
        neutral_label = "neutral_head"
        output_frames = 12
        neutral_margin = 1.0
    elif branch == "upper":
        class_order = ["Upper_Limb_Left_Rotation", "Upper_Limb_Right_Rotation", "neutral_upper_limb"]
        positive_labels = {"Upper_Limb_Left_Rotation", "Upper_Limb_Right_Rotation"}
        neutral_label = "neutral_upper_limb"
        output_frames = 12
        neutral_margin = 1.0
    elif branch == "leg":
        class_order = ["Pedaling", "neutral_leg"]
        positive_labels = {"Pedaling", "pedaling"}
        neutral_label = "neutral_leg"
        output_frames = 20
        neutral_margin = 0.0
    else:
        raise ValueError(f"unknown gesture_definition: {gesture_definition}")
    gesture_info = gesture_info.fillna("")

    skeleton_dir = Path(skeleton_data_folder)
    subset = str(datasettype)
    fold = int(fold_number)
    participants = {participant_from_name(value) for value in file_list}
    participants.discard("")
    if not participants:
        raise ValueError(f"fold {fold} {subset} file list contains no participant sessions")

    # 建立 skeleton 文件索引。
    skeleton_index = {
        normalize_clip_id(path.name): path.name
        for path in sorted(skeleton_dir.glob("*_skeleton.csv"))
    }
    valid_maneuvers = {"straight", "yield", "overtake", "overtake_left", "overtake_right"}
    clips: dict[str, dict] = {}
    for _index, row in gesture_info.iterrows():
        maneuver_label = clean_text(row.get("maneuver_label")).lower()
        clip_id = normalize_clip_id(
            row.get("generated_video_filename")
            or row.get("base_clip_id")
            or row.get("Original Filename")
        )
        if maneuver_label not in valid_maneuvers or clip_id not in skeleton_index or clip_id in clips:
            continue
        participant_id = clean_text(row.get("Participant_ID"))
        if not participant_id:
            participant_id = participant_from_name(clip_id)
        participant_id = str(int(float(participant_id))) if participant_id else ""
        if participant_id not in participants:
            continue
        clips[clip_id] = {
            "clip_id": clip_id,
            "participant_id": participant_id,
            "skeleton_file": skeleton_index[clip_id],
            "fixed_fps": parse_float(row.get("fixed_fps")),
        }
    if not clips:
        raise ValueError(f"fold {fold} {subset} has no matching scene skeletons")

    # 遍历每个手势并收集相应动作区间。
    gestures = []
    for _index, row in gesture_info.iterrows():
        clip_id = normalize_clip_id(row.get("Original Filename"))
        if clip_id not in clips:
            continue
        gesture_type = clean_text(row.get("Gesture Type"))
        use_flag = clean_text(row.get("use_flag")).lower()
        exclude_reason = clean_text(row.get("Iexclude_reason"))
        gesture_branch = branch_for_gesture(gesture_type, exclude_reason)
        if not gesture_branch and not is_excluded(use_flag):
            continue
        interval = row_interval(row, clips[clip_id]["fixed_fps"])
        if interval is None:
            continue
        gestures.append(
            {
                "clip_id": clip_id,
                "gesture_type": gesture_type,
                "branch": gesture_branch,
                "start_sec": interval[0],
                "end_sec": interval[1],
                "use_flag": use_flag,
                "exclude_reason": exclude_reason,
            }
        )

    active_by_clip: dict[str, list[dict]] = defaultdict(list)
    hard_excludes_by_clip: dict[str, list[tuple[float, float]]] = defaultdict(list)
    short_action_by_clip_branch: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in gestures:
        # 将不应进入样本的区间分到阻断列表。
        if is_excluded(row["use_flag"]):
            if row["exclude_reason"] == "duration_under_1s_hard_exclude" and row["branch"]:
                short_action_by_clip_branch[row["clip_id"]][row["branch"]].append((row["start_sec"], row["end_sec"]))
            else:
                hard_excludes_by_clip[row["clip_id"]].append((row["start_sec"], row["end_sec"]))
            continue
        if row["branch"] == branch and row["gesture_type"] in positive_labels:
            active_by_clip[row["clip_id"]].append(row)
            if row["end_sec"] - row["start_sec"] < 1.0 - 1e-9:
                short_action_by_clip_branch[row["clip_id"]][branch].append((row["start_sec"], row["end_sec"]))

    skeleton_base = skeleton_dir.relative_to(root)

    specs: dict[str, list[dict]] = {label: [] for label in class_order}
    for clip_id, clip in sorted(clips.items()):
        # 读取当前场景的骨骼时间轴。
        skeleton_data = pd.read_csv(skeleton_dir / clip["skeleton_file"])
        times = pd.to_numeric(skeleton_data.iloc[:, 0], errors="coerce").to_numpy(dtype=np.float32)
        times = np.sort(times[np.isfinite(times)])
        if times.size == 0:
            continue

        full_start, full_end = float(times[0]), float(times[-1])
        branch_active = active_by_clip.get(clip_id, [])
        branch_short = list(short_action_by_clip_branch.get(clip_id, {}).get(branch, []))
        hard_blocks = merge_intervals(hard_excludes_by_clip.get(clip_id, []))
        branch_active_intervals = [(row["start_sec"], row["end_sec"]) for row in branch_active]

        # 裁剪动作样本。
        candidate_parts: list[tuple[dict, str, float, float]] = []
        for row in branch_active:
            if row["end_sec"] - row["start_sec"] < 1.0 - 1e-9:
                continue
            label = canonical_label(row["branch"], row["gesture_type"])
            for start, end in subtract_intervals(row["start_sec"], row["end_sec"], hard_blocks):
                if end - start < 1.0 - 1e-9:
                    branch_short.append((start, end))
                else:
                    candidate_parts.append((row, label, start, end))

        branch_forbidden = merge_intervals(branch_short)
        for _row, label, start, end in candidate_parts:
            for safe_start, safe_end in subtract_intervals(start, end, branch_forbidden):
                if safe_end - safe_start >= 1.0 - 1e-9:
                    specs[label].append(
                        {
                            "label": label,
                            "clip_id": clip_id,
                            "participant_id": clip["participant_id"],
                            "skeleton_file": clip["skeleton_file"],
                            "skeleton_csv": str(skeleton_base / clip["skeleton_file"]),
                            "start_sec": safe_start,
                            "end_sec": safe_end,
                        }
                    )

        # 生成 neutral 样本。
        neutral_intervals = get_neutral_intervals(
            [start for start, _end in branch_active_intervals + branch_short],
            [end for _start, end in branch_active_intervals + branch_short],
            total_duration=full_end,
            margin=float(neutral_margin),
            blocked_intervals=hard_blocks,
            start_time=full_start,
        )
        for start, end in neutral_intervals:
            if end - start < 1.0 - 1e-9:
                continue
            specs[neutral_label].append(
                {
                    "label": neutral_label,
                    "clip_id": clip_id,
                    "participant_id": clip["participant_id"],
                    "skeleton_file": clip["skeleton_file"],
                    "skeleton_csv": str(skeleton_base / clip["skeleton_file"]),
                    "start_sec": start,
                    "end_sec": end,
                }
            )

    base_half_counts = {label: window_count(specs[label], 0.5) for label in class_order}
    balance_target = lower_median_cap(list(base_half_counts.values()))
    candidate_samples: dict[str, list[dict]] = {}
    for label in class_order:
        slide_interval = choose_slide_interval(specs[label], balance_target)
        candidate_samples[label] = []
        for spec in specs[label]:
            candidate_samples[label].extend(
                process_skeleton_file(
                    spec["skeleton_file"], spec["start_sec"], spec["end_sec"], spec["label"], skeleton_dir,
                    int(output_frames), slide_interval, spec,
                )
            )

    requested_target = formal_exact_target(root, fold, branch, subset) or balance_target
    exact_target = min([int(requested_target)] + [len(candidate_samples[label]) for label in class_order])
    samples = balance_gesture_data(candidate_samples, exact_target, class_order, fold)
    return [(sample["x"], sample["label"]) for sample in samples]


# 裁剪手势数据。
def CropGestureData(
    gesture_info_path, skeleton_data_folder, train_files, val_files, test_files, output_dir, fold_number,
    gesture_definition,
):
    # 读取 gesture_crop_info.csv。
    gesture_info = pd.read_csv(gesture_info_path, dtype=str, encoding="utf-8-sig").fillna("")

    # 处理训练、验证、测试数据。
    datasets = {}
    for subset, files in (("train", train_files), ("val", val_files), ("test", test_files)):
        datasets[subset] = process_files(files, gesture_info, skeleton_data_folder, fold_number, subset, gesture_definition)

    # 保存数据到文件。
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    branch = str(gesture_definition).strip().lower()
    if branch == "upper_limb":
        branch = "upper"
    for subset, data in datasets.items():
        with (output_path / f"fold_{fold_number}_{branch}_{subset}_data.pkl").open("wb") as handle:
            pickle.dump(data, handle)

    count_file = output_path / f"fold_{fold_number}_{branch}_data_Manuever_Type_Counts.txt"
    with count_file.open("w", encoding="utf-8") as handle:
        # 打印统计结果。
        title = f"Manuever Type Counts fold {fold_number} ({branch}):"
        print(title)
        handle.write(title + "\n")
        for subset, name in (("train", "Training"), ("val", "Validation"), ("test", "Testing")):
            print(f"{name} Set:")
            handle.write(f"{name}:\n")
            counts = Counter(label for _segment, label in datasets[subset])
            for gesture_type, count in counts.items():
                line = f"{gesture_type}: {count} segments"
                print(line)
                handle.write(line + "\n")
            total_line = f"Total segments: {len(datasets[subset])}"
            print(total_line)
            handle.write(total_line + "\n")
    print(f"Manuever Type Counts saved to {count_file.name}.")
    print(f"Data saved to {output_path} directory.")
    return datasets


project_root = Path(__file__).resolve().parents[2]
run_root = project_root.parent
gesture_info_path = project_root / "Gesture2Manuever_Prediction" / "VideoCrop" / "gesture_crop_info.csv"
skeleton_data_folder = project_root / "Full33SkeletonData" / "stage1_scene_clips"
output_dir = run_root / "release_data" / "GestureDataset"

if __name__ == "__main__":
    split_dir = project_root / "Gesture2Manuever_Prediction" / "VideoCrop"
    for fold_number in range(1, 6):
        # 加载每个 fold 的文件列表。
        with (split_dir / f"fold_{fold_number}_train_files.txt").open("r", encoding="utf-8-sig") as handle:
            train_files = [line.strip() for line in handle if line.strip()]
        with (split_dir / f"fold_{fold_number}_val_files.txt").open("r", encoding="utf-8-sig") as handle:
            val_files = [line.strip() for line in handle if line.strip()]
        with (split_dir / f"fold_{fold_number}_test_files.txt").open("r", encoding="utf-8-sig") as handle:
            test_files = [line.strip() for line in handle if line.strip()]

        for gesture_definition in ("head", "upper", "leg"):
            # 执行数据截取。
            datasets = CropGestureData(
                gesture_info_path, skeleton_data_folder, train_files, val_files, test_files,
                output_dir, fold_number, gesture_definition,
            )
            print(gesture_definition, {subset: dict(Counter(label for _x, label in data)) for subset, data in datasets.items()})
