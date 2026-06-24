import sys
import os
sys.path.append("Gesture2Manuever_Prediction")
from VideoCrop.GestureCrop import CropGestureData
from VideoCrop.ManueverCrop_oldsplit import CropManueverData
from GestureClassification.dataloader import GestureLoaddata
from GestureClassification.model_training import GestureTraining
from GestureClassification.model_evaluation import GestureEvaluation
from ManueverPrediction.dataloader_old import ManueverLoaddata
from ManueverPrediction.model_training_old import ManueverTraining
from ManueverPrediction.model_evaluation import ManueverEvaluation
# # 获取脚本所在的绝对路径，并将Gesture2Manuever_Prediction添加到sys.path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(current_dir, "Gesture2Manuever_Prediction"))
# from Gesture2Manuever_Prediction.VideoCrop.GestureCrop import CropGestureData
# from Gesture2Manuever_Prediction.VideoCrop.ManueverCrop_oldsplit import CropManueverData
# from GestureClassification.dataloader import GestureLoaddata
# from GestureClassification.model_training import GestureTraining
# from GestureClassification.model_evaluation import GestureEvaluation
# from Gesture2Manuever_Prediction.ManueverPrediction.dataloader_old import ManueverLoaddata
# from Gesture2Manuever_Prediction.ManueverPrediction.model_training_old import ManueverTraining
# from Gesture2Manuever_Prediction.ManueverPrediction.model_evaluation import ManueverEvaluation

import torch

gesture_info_path = 'Gesture2Manuever_Prediction/VideoCrop/gesture_crop_info.csv'
skeleton_data_folder = 'Full33SkeletonData' 
# segment
gesture_segment_file = 'Gesture2Manuever_Prediction/VideoCrop/all_gesture_segments.pkl'
# gesture_segment_file = 'Gesture2Manuever_Prediction/VideoCrop/all_segments.pkl'
manuever_segment_file = 'Gesture2Manuever_Prediction/VideoCrop/all_manuever_segments.pkl'
# dataloader
explicit_dataloader_save_path = 'Gesture2Manuever_Prediction/GestureClassification/LSTM_optionOut_kfold9_hidden128_layer2_batch64_NOscheduler/explicit_dataloader.pkl'
implicit_dataloader_save_path = 'Gesture2Manuever_Prediction/GestureClassification/LSTM_optionOut_kfold9_hidden128_layer2_batch64_NOscheduler/implicit_dataloader.pkl'
manuever_dataloader_save_path = 'Gesture2Manuever_Prediction/ManueverPrediction/dataloader.pkl'
# model
# explicit_model_save_path = 'Gesture2Manuever_Prediction/GestureClassification/LSTM_optionOut_kfold9_hidden128_layer2_batch64_NOscheduler/explicit_Gesture_LSTM_Model.pth'
# implicit_model_save_path = 'Gesture2Manuever_Prediction/GestureClassification/LSTM_optionOut_kfold9_hidden128_layer2_batch64_NOscheduler/implicit_Gesture_LSTM_Model.pth'
explicit_model_save_folder = 'Gesture2Manuever_Prediction/GestureClassification/LSTM_optionOut_kfold9_hidden128_layer2_batch64_NOscheduler'
implicit_model_save_folder = 'Gesture2Manuever_Prediction/GestureClassification/LSTM_optionOut_kfold9_hidden128_layer2_batch64_NOscheduler'
manuever_model_save_path = 'Gesture2Manuever_Prediction/ManueverPrediction/model.pkl'
# Label encoder
# explicit_LabelEncoder_save_path = 'Gesture2Manuever_Prediction/GestureClassification/LSTM_optionOut_kfold9_hidden128_layer2_batch64_NOscheduler/explicit_label_encoder.pkl'
# implicit_LabelEncoder_save_path = 'Gesture2Manuever_Prediction/GestureClassification/LSTM_optionOut_kfold9_hidden128_layer2_batch64_NOscheduler/implicit_label_encoder.pkl'
explicit_LabelEncoder_save_folder = 'Gesture2Manuever_Prediction/GestureClassification/LSTM_optionOut_kfold9_hidden128_layer2_batch64_NOscheduler'
implicit_LabelEncoder_save_folder = 'Gesture2Manuever_Prediction/GestureClassification/LSTM_optionOut_kfold9_hidden128_layer2_batch64_NOscheduler'
manuever_LabelEncoder_save_path = 'Gesture2Manuever_Prediction/ManueverPrediction/labelencoder.pkl'
# Result visualization: confusion matrix
# explicit_confusion_matrix_save_path = 'Gesture2Manuever_Prediction/GestureClassification/LSTM_optionOut_kfold9_hidden128_layer2_batch64_NOscheduler/explicit_Gesture_LSTM_Model.jpg'
# implicit_confusion_matrix_save_path = 'Gesture2Manuever_Prediction/GestureClassification/LSTM_optionOut_kfold9_hidden128_layer2_batch64_NOscheduler/implicit_Gesture_LSTM_Model.jpg'
explicit_confusion_matrix_save_folder = 'Gesture2Manuever_Prediction/GestureClassification/LSTM_optionOut_kfold9_hidden128_layer2_batch64_NOscheduler'
implicit_confusion_matrix_save_folder = 'Gesture2Manuever_Prediction/GestureClassification/LSTM_optionOut_kfold9_hidden128_layer2_batch64_NOscheduler'
manuever_confusion_matrix_save_path = 'Gesture2Manuever_Prediction/ManueverPrediction/confusion_matrix.jpg'
# Dataloader of best model
# explicit_best_model_dataloader_save_path = 'Gesture2Manuever_Prediction/GestureClassification/LSTM_optionOut_kfold9_hidden128_layer2_batch64_NOscheduler/explicit_Bestmodel_dataloader.pkl'
# implicit_best_model_dataloader_save_path = 'Gesture2Manuever_Prediction/GestureClassification/LSTM_optionOut_kfold9_hidden128_layer2_batch64_NOscheduler/implicit_Bestmodel_dataloader.pkl'
# manuever_best_model_dataloader_save_path = 'Gesture2Manuever_Prediction/ManueverPrediction/best_model_dataloader.pkl'



#########################
# Gesture Classification
#
#########################
print('Belowing serves for the gesture classfication:')
num_epochs_explicit = 40
num_epochs_implicit = 200

# Crop the time-series skeleton data according to gesture 
if not os.path.exists(gesture_segment_file):
    print('Data not segmented yet, start with segmenting now...')
    CropGestureData(gesture_info_path,skeleton_data_folder,gesture_segment_file)
else:
    print('Data already segmented.')

# checking if GPU is available
device = torch.device("cpu")
if (torch.cuda.is_available()):
    device = torch.device("cuda:0")
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')

# Prepare the dataloader
if not os.path.exists(explicit_dataloader_save_path) and not os.path.exists(implicit_dataloader_save_path):
    print('Dataloader not prepared yet, start with preparing dataloader now...')
    GestureLoaddata(gesture_segment_file,device,explicit_dataloader_save_path,implicit_dataloader_save_path)
else:
    print('Dataloader already prepared.')

# Train a LSTM model and save it
# if not os.path.exists(explicit_model_save_folder):
print('Explicit classification model not trained yet, start with model training now...')
GestureTraining(explicit_dataloader_save_path,explicit_model_save_folder,explicit_LabelEncoder_save_folder,
            explicit_confusion_matrix_save_folder,num_epochs_explicit)
# else:
#     print(f"Explicit classification model existed in {explicit_model_save_folder}")
    
#     GestureEvaluation(explicit_model_save_folder,explicit_confusion_matrix_save_folder)

# if not os.path.exists(implicit_model_save_folder):
print('Implicit classification model not trained yet, start with model training now...')
GestureTraining(implicit_dataloader_save_path,implicit_model_save_folder,implicit_LabelEncoder_save_folder,
            implicit_confusion_matrix_save_folder,num_epochs_implicit)
# else:
#     print(f"Implicit classification model existed in {implicit_model_save_folder}")
    
#     GestureEvaluation(implicit_model_save_folder,implicit_confusion_matrix_save_folder)


# #########################
# # Manuever Prediction
# #
# #########################
# print('Belowing serves for the manuever prediction:')
# num_epochs_manuever = 80

# # Crop the time-series skeleton data according to gesture 
# if not os.path.exists(manuever_segment_file):
#     print('Data not segmented yet, start with segmenting now...')
#     CropManueverData(gesture_info_path,skeleton_data_folder,manuever_segment_file)
# else:
#     print('Data already segmented.')

# # checking if GPU is available
# device = torch.device("cpu")
# if (torch.cuda.is_available()):
#     device = torch.device("cuda:0")
#     print('Training on GPU.')
# else:
#     print('No GPU available, training on CPU.')

# # Prepare the dataloader
# if not os.path.exists(manuever_dataloader_save_path):
#     print('Dataloader not prepared yet, start with preparing dataloader now...')
#     ManueverLoaddata(manuever_segment_file,device,manuever_dataloader_save_path)
# else:
#     print('Dataloader already prepared.')

# # Train a LSTM model and save it
# if not os.path.exists(manuever_model_save_path):
#     print('Manuever Prediction model not trained yet, start with model training now...')
#     ManueverTraining(manuever_dataloader_save_path,manuever_model_save_path,manuever_LabelEncoder_save_path,
#                      manuever_confusion_matrix_save_path,manuever_best_model_dataloader_save_path,num_epochs_manuever)
# else:
#     print(f"Manuever prediction model existed in {manuever_model_save_path}")
    
#     ManueverEvaluation(manuever_model_save_path,manuever_best_model_dataloader_save_path,manuever_confusion_matrix_save_path)

