import os
import sys
import time
import argparse

sys.path.append("Gesture2Manuever_Prediction")
from ManueverPrediction_Combined.dataloader import create_and_save_dataloaders
from ManueverPrediction_Combined.model_training import train_all_folds

def parse_arguments():
    parser = argparse.ArgumentParser(description="Specify model classification")
    parser.add_argument("--model_type", type=str, choices=["weighting_NN", "weighting_set"], required=True, help="First layer of classification")
    parser.add_argument("--model_subtype", type=str, choices=["Ex_Im", "Ex_only"], required=True, help="Second layer of classification")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Assign variables based on arguments
    model_type = args.model_type
    model_subtype = args.model_subtype

    data_dir = 'Gesture2Manuever_Prediction/ManueverPrediction_Combined/ManueverDataset'
    # output_dir = 'Gesture2Manuever_Prediction/ManueverPrediction_Combined/DataLoaders'
    # dataloaders_dir = 'Gesture2Manuever_Prediction/ManueverPrediction_Combined/DataLoaders'  # Path to the folder containing dataloader .pkl files

    if model_type == "weighting_NN":
        if model_subtype == "Ex_Im":
            model_save_dir = 'Gesture2Manuever_Prediction/ManueverPrediction_Combined/Models_WeightingNN_ExplicitImplicit'
        elif model_subtype == "Ex_only":
            model_save_dir = 'Gesture2Manuever_Prediction/ManueverPrediction_Combined/Models_WeightingNN_ExplicitOnly'
    elif model_type == "weighting_set":
        if model_subtype == "Ex_Im":
            model_save_dir = 'Gesture2Manuever_Prediction/ManueverPrediction_Combined/Models_ExplicitImplicit'
        elif model_subtype == "Ex_only":
            model_save_dir = 'Gesture2Manuever_Prediction/ManueverPrediction_Combined/Models_ExplicitOnly'
    # model_save_dir = 'Gesture2Manuever_Prediction/ManueverPrediction_Combined/Models_ExplicitImplicit'
    for fold_number in range(1, 6):
        gesture_model_explicit_dataloaders_path = f'Gesture2Manuever_Prediction/GestureClassification/ExplicitModels/dataloader_fold_{fold_number}.pkl'
        gesture_model_implicit_dataloaders_path = f'Gesture2Manuever_Prediction/GestureClassification/ImplicitModels/dataloader_fold_{fold_number}.pkl'
        gesture_model_explicit_path = f'Gesture2Manuever_Prediction/GestureClassification/ExplicitModels/best_model_fold_{fold_number}.pkl'
        gesture_model_implicit_path = f'Gesture2Manuever_Prediction/GestureClassification/ImplicitModels/best_model_fold_{fold_number}.pkl'
        dataloaders = create_and_save_dataloaders(data_dir, fold_number)
        start_time = time.time()  # Start timing
        train_all_folds(model_type, model_subtype, dataloaders, model_save_dir,gesture_model_explicit_dataloaders_path, gesture_model_implicit_dataloaders_path,
                        gesture_model_explicit_path, gesture_model_implicit_path, fold_number)
        end_time = time.time()  # End timing
        total_time = end_time - start_time
        print(f"Total training time for all folds: {total_time:.2f} seconds")
        training_time_save_path = os.path.join(model_save_dir,f'training_time_fold_{fold_number}.txt')
        with open(training_time_save_path, 'w') as f:
            f.write(f"Total training time for fold {fold_number}: {total_time:.2f} seconds")

