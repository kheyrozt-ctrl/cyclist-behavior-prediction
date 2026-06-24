VideoCrop contains fold split files, label tables, blocker lists, and data construction scripts.
VideoCrop 存放 fold 划分文件、标签表、blocker 列表和数据生成脚本。

Raw video cropping is not included in this release because the pipeline starts from skeleton CSV files.
本文件夹不包含原始视频裁剪，因为流程从骨架 CSV 文件开始。

`gesture_crop_info.csv` is the main Stage1 action and scene metadata table.
`gesture_crop_info.csv` 是主要的 Stage1 动作区间和 scene 元数据表。

It is expected to validate as 2093 rows, 466 normalized exclude/disable rows, and 243 affected clips.
它应校验为 2093 行、466 条归一化 exclude/disable 行、243 个受影响 clip。

`fold_{1..5}_{train,val,test}_files.txt` defines the Stage1 fold split.
`fold_{1..5}_{train,val,test}_files.txt` 定义 Stage1 的 fold 划分。

`stage1_exact_targets.txt` stores the exact-balanced Stage1 per-class caps.
`stage1_exact_targets.txt` 存放 Stage1 每类 balanced 目标数量。

`stage2_excluded_18_baseclips.txt` stores the 18 whole-clip Stage2 blockers.
`stage2_excluded_18_baseclips.txt` 存放 Stage2 需要整段排除的 18 个 base clip。

`GestureCrop_newsplit.py` builds Stage1 head / upper-limb / leg pkl data.
`GestureCrop_newsplit.py` 生成 Stage1 头部、上肢、腿部 pkl 数据。

`ManueverCrop_combine.py` builds Stage2 raw 120 x 66 maneuver pkl data.
`ManueverCrop_combine.py` 生成 Stage2 原始 120 x 66 maneuver pkl 数据。
