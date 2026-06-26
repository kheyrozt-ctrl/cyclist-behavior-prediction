Gesture2Manuever_Prediction contains the full two-stage prediction pipeline.
Gesture2Manuever_Prediction 存放完整两阶段预测流程。

`VideoCrop/` builds Stage1 and Stage2 pkl data from skeleton CSV files and label tables.
`VideoCrop/` 从骨架 CSV 和标签表生成 Stage1 与 Stage2 pkl 数据。

`GestureClassification/` trains the Stage1 head / upper-limb / leg classifiers.
`GestureClassification/` 训练 Stage1 头部、上肢、腿部分类器。

`ManueverPrediction_Combined/` trains the Stage2 combined maneuver classifier.
`ManueverPrediction_Combined/` 训练 Stage2 组合机动行为分类器。

Run the four pipeline steps from the release root folder.
请根目录运行四个流程步骤。  详情间md文件
