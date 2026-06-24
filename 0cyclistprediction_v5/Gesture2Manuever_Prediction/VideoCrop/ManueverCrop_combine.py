from __future__ import annotations

import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


# 函数：处理数据生成。
def process_files(file_list, data, gesture_info, skeleton_data_folder, fold_number=1, datasettype="train"):
    project_root = Path(__file__).resolve().parents[2]
    window_size = 120
    target_fps = 12.0
    step_size = {"straight": 2, "yield": 1, "overtake": 1}
    excluded_route_anchors = {3, 5}

    def normalize_label(label):
        return str(label).strip().replace("_left", "").replace("_right", "")

    def clip_key(value):
        name = Path(str(value)).stem.strip()
        if name.endswith("_skeleton"):
            name = name[: -len("_skeleton")]
        if name.endswith("_cam2rgbfixed"):
            name = name[: -len("_cam2rgbfixed")]
        return name.lower()

    def route_anchor(value):
        return None if pd.isna(value) or str(value).strip() == "" else int(float(value))

    # 读取对应的骨骼数据文件，并重采样到目标帧率。
    def read_skeleton(row):
        key = clip_key(row.base_clip_id)
        if key not in skeleton_index:
            raise ValueError(f"{row.base_clip_id} is missing from gesture_crop_info.csv")
        frame = pd.read_csv(skeleton_index[key]).rename(columns={"timestamp": "timestamp_s"})
        # 只保留 66 列姿态特征。
        columns = [column for column in frame.columns if column.endswith("_x") or column.endswith("_y")]
        if len(columns) != 66:
            raise ValueError(f"expected 66 pose columns ending with _x/_y, got {len(columns)}")

        coords_df = frame[columns].apply(pd.to_numeric, errors="coerce")
        coords_df = coords_df.interpolate(method="linear", limit_direction="both").ffill().bfill().fillna(0.0)
        coords = coords_df.to_numpy(dtype="float32")
        timestamps = pd.to_numeric(frame["timestamp_s"], errors="coerce").to_numpy(dtype="float64")
        valid = np.isfinite(timestamps)
        if not valid.any():
            raise ValueError("skeleton timestamp column has no valid values")

        timestamps, coords = timestamps[valid], coords[valid]
        order = np.argsort(timestamps)
        timestamps, coords = timestamps[order], coords[order]
        timestamps, unique_idx = np.unique(timestamps, return_index=True)
        coords = coords[unique_idx]
        if len(coords) <= 1:
            return coords.astype("float32")

        duration = float(timestamps[-1] - timestamps[0])
        if duration <= 0:
            return coords[:1].astype("float32")

        frame_count = int(np.floor(duration * target_fps)) + 1
        grid = timestamps[0] + (np.arange(frame_count, dtype="float64") / target_fps)
        output = np.empty((frame_count, coords.shape[1]), dtype="float32")
        for column in range(coords.shape[1]):
            output[:, column] = np.interp(grid, timestamps, coords[:, column]).astype("float32")
        return output

    fold_data_path = (
        project_root
        / "Gesture2Manuever_Prediction"
        / "ManueverPrediction_Combined"
        / "ManueverDataset"
        / f"fold_{fold_number}_{datasettype}_data.csv"
    )
    fold_data = pd.read_csv(fold_data_path).fillna("")

    # 获取当前文件的相关 gesture/scene 信息。
    info = gesture_info.fillna("").copy()
    info["scene_key"] = info["base_clip_id"].map(clip_key)
    keep_columns = [
        "scene_key",
        "base_clip_id",
        "generated_video_filename",
        "skeleton_csv",
        "route_anchor",
        "maneuver_label",
        "official_fixed_frame_start",
        "official_fixed_frame_end",
        "official_fixed_sec_start",
        "official_fixed_sec_end",
    ]
    scene_info = info[[column for column in keep_columns if column in info.columns]].drop_duplicates("scene_key")

    fold_data["scene_key"] = fold_data["base_clip_id"].map(clip_key)
    fold_data = fold_data[["scene_key"]].drop_duplicates().merge(scene_info, on="scene_key", how="left")
    fold_data = fold_data.drop(columns=["scene_key"])

    skeleton_folder = Path(skeleton_data_folder)
    if not skeleton_folder.is_absolute():
        skeleton_folder = project_root / skeleton_folder
    skeleton_files = {}
    for path in skeleton_folder.glob("*_skeleton.csv"):
        skeleton_files.setdefault(clip_key(path.stem), []).append(path)

    # 获取唯一的骨骼文件名。
    skeleton_index = {}
    for column in [column for column in ("Filename", "Original Filename") if column in gesture_info.columns]:
        for value in gesture_info[column].dropna().unique():
            matches = skeleton_files.get(clip_key(value), [])
            if len(matches) == 1:
                skeleton_index[clip_key(value)] = matches[0]
            elif len(matches) > 1:
                raise ValueError(f"multiple skeleton files match {value}")

    blocker_path = project_root / "Gesture2Manuever_Prediction" / "VideoCrop" / "stage2_excluded_18_baseclips.txt"
    stage2_blockers = {line.strip() for line in blocker_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()}
    allowed_participants = {
        match.group(1)
        for value in file_list
        if (match := re.match(r"^Participant_0*(\d+)_", Path(str(value)).stem)) is not None
    }

    for row in fold_data.itertuples(index=False):
        participant = re.match(r"^Participant_0*(\d+)_", str(row.base_clip_id))
        if allowed_participants and (participant is None or participant.group(1) not in allowed_participants):
            continue
        if str(row.base_clip_id) in stage2_blockers:
            continue

        # 判断 maneuver_type。
        maneuver_type = normalize_label(row.maneuver_label)
        if route_anchor(getattr(row, "route_anchor", None)) in excluded_route_anchors:
            continue
        if maneuver_type not in step_size:
            raise ValueError(f"missing step size for {maneuver_type}")

        skeleton_data = read_skeleton(row)
        if len(skeleton_data) < window_size:
            continue

        # 滑动窗口截取数据。
        for start_idx in range(0, len(skeleton_data) - window_size + 1, max(1, step_size[maneuver_type])):
            # 添加到数据集中。
            data.append((skeleton_data[start_idx : start_idx + window_size].astype("float32"), maneuver_type))

    return data


def CropManueverData(gesture_info_path, skeleton_data_folder, train_files, val_files, test_files, output_dir, fold_number):
    # 读取 gesture_crop_info.csv。
    gesture_info = pd.read_csv(gesture_info_path)
    # 定义存储所有 segments 的列表。
    train_data, val_data, test_data = [], [], []

    # 处理训练、验证、测试数据。
    train_data = process_files(train_files, train_data, gesture_info, skeleton_data_folder, fold_number, "train")
    val_data = process_files(val_files, val_data, gesture_info, skeleton_data_folder, fold_number, "val")
    test_data = process_files(test_files, test_data, gesture_info, skeleton_data_folder, fold_number, "test")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # 统计每种 maneuver_type 的数量。
    with (output / f"fold_{fold_number}_data_Manuever_Type_Counts.txt").open("w", encoding="utf-8") as handle:
        print("Manuever Type Counts:")
        handle.write("Manuever Type Counts:\n")
        for dataset, name in [(train_data, "Training"), (val_data, "Validation"), (test_data, "Testing")]:
            maneuver_counts = Counter([maneuver_type for _, maneuver_type in dataset])
            print(f"{name} Set:")
            handle.write(f"{name}:\n")
            for maneuver_type, count in maneuver_counts.items():
                print(f"{maneuver_type}: {count} segments")
                handle.write(f"{maneuver_type}: {count} segments\n")
    print(f"Manuever Type Counts saved to fold_{fold_number}_data_Manuever_Type_Counts.txt.")

    # 保存数据到文件。
    with (output / f"fold_{fold_number}_train_data.pkl").open("wb") as handle:
        pickle.dump(train_data, handle)
    with (output / f"fold_{fold_number}_val_data.pkl").open("wb") as handle:
        pickle.dump(val_data, handle)
    with (output / f"fold_{fold_number}_test_data.pkl").open("wb") as handle:
        pickle.dump(test_data, handle)

    print(f"Data saved to {output} directory.")
    return train_data, val_data, test_data


project_root = Path(__file__).resolve().parents[2]
run_root = project_root.parent
gesture_info_path = project_root / "Gesture2Manuever_Prediction" / "VideoCrop" / "gesture_crop_info.csv"
skeleton_data_folder = project_root / "Full33SkeletonData" / "stage1_scene_clips_20260622"
output_dir = run_root / "release_data" / "ManueverDataset"

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

        # 执行数据截取。
        CropManueverData(gesture_info_path, skeleton_data_folder, train_files, val_files, test_files, output_dir, fold_number)
