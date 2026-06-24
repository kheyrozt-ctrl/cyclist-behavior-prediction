ManueverDataset stores Stage2 fold/subset split CSV files.
ManueverDataset 存放 Stage2 fold/subset 划分 CSV 文件。

The files are named `fold_{1..5}_{train,val,test}_data.csv`.
文件命名为 `fold_{1..5}_{train,val,test}_data.csv`。

These CSV files define which scene/base-clip skeletons belong to each fold and subset.
这些 CSV 定义每个 scene/base-clip 骨架属于哪个 fold 和 subset。

They are used by `../../VideoCrop/ManueverCrop_combine.py` to build Stage2 pkl windows.
它们由 `../../VideoCrop/ManueverCrop_combine.py` 用于生成 Stage2 pkl 窗口。


