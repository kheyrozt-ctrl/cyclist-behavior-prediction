import os
import sys
import time
sys.path.append("Gesture2Manuever_Prediction")
from GestureClassification.newsplit_dataloader import create_and_save_dataloaders
from GestureClassification.newsplit_model_training import train_all_folds
from GestureClassification.newsplit_exmodel_definition import ExplicitLSTMClassifier
from GestureClassification.newsplit_immodel_definition import ImplicitLSTMClassifier


if __name__ == "__main__":

    data_dir_explicit = 'Gesture2Manuever_Prediction/GestureClassification/ExplicitGestureDataset'
    data_dir_implicit = 'Gesture2Manuever_Prediction/GestureClassification/ImplicitGestureDataset'

    model_save_dir_explicit = 'Gesture2Manuever_Prediction/GestureClassification/ExplicitModels'
    model_save_dir_implicit = 'Gesture2Manuever_Prediction/GestureClassification/ImplicitModels'

    for fold_number in range(1, 6):
        dataloaders_explicit = create_and_save_dataloaders(data_dir_explicit, 'explicit', fold_number)
        dataloaders_implicit = create_and_save_dataloaders(data_dir_implicit, 'implicit', fold_number)

        start_time = time.time()  # Start timing
        os.makedirs(model_save_dir_explicit,exist_ok=True)
        train_all_folds(dataloaders_explicit, model_save_dir_explicit, ExplicitLSTMClassifier, fold_number)
        end_time = time.time()  # End timing
        total_time = end_time - start_time
        print(f"Explicit training time for fold {fold_number}: {total_time:.2f} seconds")
        training_time_save_path = os.path.join(model_save_dir_explicit,f'training_time.txt')
        with open(training_time_save_path, 'w') as f:
            f.write(f"Explicit training time for fold {fold_number}: {total_time:.2f} seconds")


        start_time = time.time()  # Start timing
        os.makedirs(model_save_dir_implicit,exist_ok=True)
        train_all_folds(dataloaders_implicit, model_save_dir_implicit, ImplicitLSTMClassifier, fold_number)
        end_time = time.time()  # End timing
        total_time = end_time - start_time
        print(f"Implicit training time for fold {fold_number}: {total_time:.2f} seconds")
        training_time_save_path = os.path.join(model_save_dir_implicit,f'training_time.txt')
        with open(training_time_save_path, 'w') as f:
            f.write(f"Implicit training time for fold {fold_number}: {total_time:.2f} seconds")

