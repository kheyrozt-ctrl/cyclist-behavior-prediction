# Cyclist Prediction
# 骑行者预测

> Public repository note: raw-data filenames, dated session manifests, local
> split lists, and generated label tables are intentionally excluded. Use the
> anonymized Hugging Face dataset and `../tools/prepare_hf_stage2.py` for public
> Stage2 data preparation. The legacy builders below remain for authorized
> local-source reconstruction only.

本版基于 Zheng Tian-Ana 骨架数据的两阶段骑行者预测流程和代码。添加了部分数据排除规则，修改了场景动作映射和 Stage2 等内容。
This version is based on the Zheng Tian-Ana skeleton-data two-stage cyclist prediction pipeline and code. Some data exclusion rules, scene-action mappings, and Stage2 components were modified.

可以从已经准备好的骨架 CSV 和标签表开始运行，不包含原始视频。
It can run from prepared skeleton CSV files and label tables. Raw videos are not included.


## 运行依赖
## Requirements

建议使用 Python 3.10 包。
Python 3.10 is recommended.

```text
numpy
pandas
scikit-learn
torch
tqdm
```

## 运行顺序
## Run Order

根目录下运行命令。
Run the commands from the root directory.

第 1 步生成 Stage1 gesture 窗口。
Step 1 generates Stage1 gesture windows.

```bash
python Gesture2Manuever_Prediction/VideoCrop/GestureCrop_newsplit.py
```

第 2 步训练 Stage1 的头部、上肢、腿部分支分类器。
Step 2 trains the Stage1 head, upper-limb, and leg branch classifiers.

```bash
python Gesture2Manuever_Prediction/GestureClassification/main.py
```

第 3 步生成 Stage2 maneuver 窗口。
Step 3 generates Stage2 maneuver windows.

```bash
python Gesture2Manuever_Prediction/VideoCrop/ManueverCrop_combine.py
```

第 4 步训练 Stage2 组合机动行为分类器。
Step 4 trains the Stage2 combined maneuver classifier.

```bash
python Gesture2Manuever_Prediction/ManueverPrediction_Combined/main.py
```

默认情况下，上述命令会运行 fold 1-5。
By default, the commands above run folds 1-5.

如果只想运行单个 fold，可以添加 `--fold 1`、`--fold 2` 等参数。
To run only one fold, add `--fold 1`, `--fold 2`, and so on.


`numpy` 用于数值数组和窗口处理。
`numpy` is used for numerical arrays and window processing.

`pandas` 用于读取和写入标签、划分和骨架 CSV 文件。
`pandas` is used for reading and writing label, split, and skeleton CSV files.

`scikit-learn` 用于指标计算、标签编码和标准化。
`scikit-learn` is used for metric calculation, label encoding, and standardization.

`torch` 用于所有 BiLSTM 模型和训练。
`torch` is used for all BiLSTM models and training.

`tqdm` 用于数据生成和训练过程中的进度显示。
`tqdm` is used for progress display during data generation and training.

## 文件夹说明
## Folder Guide

`Full33SkeletonData/` 存放完整 33 点骨架 CSV 文件。一个基于完整原视频（包含六个场景），一个基于裁剪过的场景片段。
`Full33SkeletonData/` stores full 33-point skeleton CSV files. One set is based on full original videos with six scenes, and one set is based on cropped scene clips.

`Full33SkeletonData/stage1_scene_clips/` stores the authorized local
scene/base-clip skeleton CSV files used for Stage1 and Stage2 construction.

`Gesture2Manuever_Prediction/` 存放所有数据生成和模型训练代码。
`Gesture2Manuever_Prediction/` stores all data generation and model training code.

`Gesture2Manuever_Prediction/VideoCrop/` 存放 fold 划分 txt、标签表、balanced target、禁用/排除列表和数据生成脚本。
`Gesture2Manuever_Prediction/VideoCrop/` stores fold split txt files, label tables, balanced targets, disable/exclude lists, and data generation scripts.

`Gesture2Manuever_Prediction/GestureClassification/` 存放 Stage1 的头部、上肢、腿部动作分类器。
`Gesture2Manuever_Prediction/GestureClassification/` stores the Stage1 head, upper-limb, and leg action classifiers.

`Gesture2Manuever_Prediction/GestureClassification/GestureDataset/` 是运行数据目录。
`Gesture2Manuever_Prediction/GestureClassification/GestureDataset/` is the runtime data directory.

`Gesture2Manuever_Prediction/ManueverPrediction_Combined/` 存放 Stage2 组合机动行为分类器。
`Gesture2Manuever_Prediction/ManueverPrediction_Combined/` stores the Stage2 combined maneuver classifier.

`Gesture2Manuever_Prediction/ManueverPrediction_Combined/ManueverDataset/` is
a placeholder for locally generated Stage2 inputs. Generated files are ignored.

## Public Data Inputs

Source-derived label tables, fold lists, blocker lists, original filenames,
recording dates, and internal paths are private release inputs and are not
included in the public Git repository.

The canonical public input is the anonymized Hugging Face dataset. It contains:

- participant-disjoint assignments for all five folds;
- anonymized Stage1 intervals in `annotations.parquet`;
- clip-level labels and eligibility in `sequences.csv`;
- 66-coordinate skeleton frames in split Parquet files.

Run `python tools/prepare_hf_stage2.py --fold 1` from the repository root to
generate public Stage2 PKL inputs. See `docs/PUBLIC_TRAINING.md`.

## 第一阶段逻辑
## Stage1 Logic

Stage1 训练三个分支分类器：头部、上肢和腿部。
Stage1 trains three branch classifiers: head, upper-limb, and leg.

Stage1 使用所有 route anchor，不排除 `route_anchor` 3 或 5（十字路口场景，包含大量动作）。
Stage1 uses all route anchors and does not exclude `route_anchor` 3 or 5 (intersection scenes with many actions).

标签 `exclude`、`disable`、`disabled` 按同一种 blocker 含义处理，不同次更新造成，暂时未统一。
The labels `exclude`, `disable`, and `disabled` are treated as the same blocker meaning; different updates caused temporary naming inconsistency.

exclude/disable 区间会阻断 Stage1 positive 和 neutral 窗口，但不会删除整条视频，影响较小的仍被第二阶段利用。
Exclude/disable intervals block Stage1 positive and neutral windows, but do not remove whole videos; clips with limited impact are still used by Stage2.

短于 1 秒的 active action 不进入 positive，只阻断其对应分支的 neutral 窗口，使得 neutral 更干净。
Active actions shorter than 1 second do not enter positive windows and only block neutral windows for their own branch, making neutral cleaner.

hard-exclude 切割后不足 1 秒的 positive 或 neutral remainder 会被丢弃，同导师之前版本逻辑。
Positive or neutral remainders shorter than 1 second after hard-exclude cutting are discarded, following the previous mentor-version logic.

Stage1 窗口由时间戳生成，并重采样到对应分支输入长度。
Stage1 windows are generated from timestamps and resampled to the corresponding branch input length.

## 第二阶段逻辑
## Stage2 Logic

Stage2 预测 `straight`、`yield`、`overtake`。
Stage2 predicts `straight`, `yield`, and `overtake`.

Stage2 只根据 `route_anchor` 列排除官方 `route_anchor in {3,5}` 的数据。
Stage2 excludes data with official `route_anchor in {3,5}` only according to the `route_anchor` column.

Stage2 uses only clips marked `stage2_eligible` in the public dataset.

Stage2 dense window 使用固定 12 Hz 重采样、10 秒、120 帧。
Stage2 dense windows use fixed 12 Hz resampling, 10 seconds, and 120 frames.

类别步长为 `straight/yield/overtake = 2/1/1`。
The class step frames are `straight/yield/overtake = 2/1/1`.

Stage2 pkl 文件保存原始 `120 x 66` 骨架窗口。
Stage2 pkl files store raw `120 x 66` skeleton windows.

在 `CombinedModel.forward(raw_skeleton_120x66)` 中，会调用冻结的 head / upper / leg Stage1 模型，并使用 `return_hidden=True`。
In `CombinedModel.forward(raw_skeleton_120x66)`, frozen head / upper / leg Stage1 models are called with `return_hidden=True`.

根据建议目前，hidden feature 会拼接为 `head 128D + upper 192D + leg 192D = 512D`。
Currently, based on the recommendation, hidden features are concatenated as `head 128D + upper 192D + leg 192D = 512D`.

512D 特征会被标准化、投影到 16D，再输入 Stage2 BiLSTM 分类。
The 512D features are standardized, projected to 16D, and then input to the Stage2 BiLSTM classifier.

## 预期输出目录
## Expected Output Folders

Stage1 生成的 pkl 数据默认写入 `../release_data/GestureDataset/`。
Stage1 generated pkl data is written to `../release_data/GestureDataset/` by default.

Stage2 生成的 pkl 数据默认写入 `../release_data/ManueverDataset/`。
Stage2 generated pkl data is written to `../release_data/ManueverDataset/` by default.

训练 checkpoint 和结果摘要默认写入 `../release_runs/`。
Training checkpoints and result summaries are written to `../release_runs/` by default.

这些输出目录会在运行时生成。
These output directories are generated at runtime.

## 注意事项
## Notes

不要从文件名里的 `S3` 或 `S5` 判断 scene 3 或 scene 5，S 仅代表顺序，且根据原模拟实验路径不同实际场景排列会改变。
Do not judge scene 3 or scene 5 from `S3` or `S5` in file names. `S` only means order, and the actual scene order changes with the original simulation route.

在本版中，scene 3 和 scene 5 始终表示 `route_anchor = 3` 或 `route_anchor = 5`。
In this version, scene 3 and scene 5 always mean `route_anchor = 3` or `route_anchor = 5`.
