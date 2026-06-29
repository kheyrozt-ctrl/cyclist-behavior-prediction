from dataclasses import dataclass


@dataclass
class GestureConfig:
    num_epochs: int = 80
    hidden_size: int = 128
    learning_rate: float = 0.001
    num_layers: int = 4
    batch_size: int = 16
    weight_decay: float = 1e-5
    patience: int = 40
    dropout: float = 0.5
    window_size: int = 30


@dataclass
class ManeuverConfig:
    num_epochs: int = 30
    hidden_size: int = 128
    learning_rate: float = 0.0001
    num_layers: int = 2
    batch_size: int = 16
    weight_decay: float = 1e-5
    dropout: float = 0.5


@dataclass
class Paths:
    gesture_data_explicit: str = "Gesture2Manuever_Prediction/GestureClassification/ExplicitGestureDataset"
    gesture_data_implicit: str = "Gesture2Manuever_Prediction/GestureClassification/ImplicitGestureDataset"
    maneuver_data: str = "Gesture2Manuever_Prediction/ManueverPrediction_Combined/ManueverDataset"
    explicit_models: str = "Gesture2Manuever_Prediction/GestureClassification/ExplicitModels"
    implicit_models: str = "Gesture2Manuever_Prediction/GestureClassification/ImplicitModels"
