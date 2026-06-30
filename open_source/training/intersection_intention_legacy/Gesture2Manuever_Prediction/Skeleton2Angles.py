import pandas as pd
import numpy as np

def calculate_angles(points_dict):
    """
    :param points_dict: dictionary of skeleton points
    :return:
    """
    df = pd.DataFrame(points_dict, index=[0])
    df_angles = pd.DataFrame()
    ####################
    # Arms angles
    ####################
    # get indices for necessary body parts
    left_shoulder_idx = "LEFT_SHOULDER_"
    left_shoulder_indices = [left_shoulder_idx + axis for axis in ["x", "y", "z"]]
    left_elbow_idx = "LEFT_ELBOW_"
    left_elbow_indices = [left_elbow_idx + axis for axis in ["x", "y", "z"]]
    left_wrist_idx = "LEFT_WRIST_"
    left_wrist_indices = [left_wrist_idx + axis for axis in ["x", "y", "z"]]
    right_shoulder_idx = "RIGHT_SHOULDER_"
    right_shoulder_indices = [right_shoulder_idx + axis for axis in ["x", "y", "z"]]
    right_elbow_idx = "RIGHT_ELBOW_"
    right_elbow_indices = [right_elbow_idx + axis for axis in ["x", "y", "z"]]
    right_wrist_idx = "RIGHT_WRIST_"
    right_wrist_indices = [right_wrist_idx + axis for axis in ["x", "y", "z"]]
    # calculate overall arm angles
    df_angles["left_arm_angle_lat"] = np.arctan2(df[left_wrist_idx + "y"] - df[left_shoulder_idx + "y"],
                                          df[left_wrist_idx + "x"] - df[left_shoulder_idx + "x"]) * 180 / np.pi
    df_angles["left_arm_angle_lon"] = np.arctan2(df[left_wrist_idx + "y"] - df[left_shoulder_idx + "y"],
                                          df[left_wrist_idx + "z"] - df[left_shoulder_idx + "z"]) * -180 / np.pi
    df_angles["right_arm_angle_lat"] = np.arctan2(df[right_wrist_idx + "y"] - df[right_shoulder_idx + "y"],
                                           df[right_wrist_idx + "x"] - df[right_shoulder_idx + "x"]) * -180 / np.pi
    df_angles["right_arm_angle_lon"] = np.arctan2(df[right_shoulder_idx + "y"] - df[right_wrist_idx + "y"],
                                           df[right_shoulder_idx + "z"] - df[right_wrist_idx + "z"]) * 180 / np.pi
    # calculate left elbow angle
    left_upper_arm_vector = pd.DataFrame(df[left_elbow_indices].values - df[left_shoulder_indices].values, columns=["dx", "dy", "dz"])
    left_forearm_vector = pd.DataFrame(df[left_wrist_indices].values - df[left_elbow_indices].values, columns=["dx", "dy", "dz"])
    lu_mag = np.sqrt(np.sum(left_upper_arm_vector.multiply(left_upper_arm_vector, axis=1), axis=1))
    lf_mag = np.sqrt(np.sum(left_forearm_vector.multiply(left_forearm_vector, axis=1), axis=1))
    df_angles["left_elbow_angle"] = (np.arccos(np.sum(left_upper_arm_vector*left_forearm_vector, axis=1) / (lu_mag*lf_mag)) * 180 / np.pi).tolist()
    # calculate right elbow angle
    right_upper_arm_vector = pd.DataFrame(df[right_elbow_indices].values - df[right_shoulder_indices].values, columns=["dx", "dy", "dz"])
    right_forearm_vector = pd.DataFrame(df[right_wrist_indices].values - df[right_elbow_indices].values, columns=["dx", "dy", "dz"])
    ru_mag = np.sqrt(np.sum(right_upper_arm_vector.multiply(right_upper_arm_vector, axis=1), axis=1))
    rf_mag = np.sqrt(np.sum(right_forearm_vector.multiply(right_forearm_vector, axis=1), axis=1))
    df_angles["right_elbow_angle"] = (np.arccos(np.sum(right_upper_arm_vector*right_forearm_vector, axis=1) / (ru_mag*rf_mag)) * 180 / np.pi).tolist()
    
    
    ####################
    # Twist angles
    ####################
    # get indices for necessary body parts
    left_shoulder_idx = "LEFT_SHOULDER_"
    right_shoulder_idx = "RIGHT_SHOULDER_"
    left_ear_idx = "LEFT_EAR_"
    right_ear_idx = "RIGHT_EAR_"
    
    # calculate overall arm angles
    df_angles["shoulder_twist_angle"] = np.arctan2(df[left_shoulder_idx + "x"] - df[right_shoulder_idx + "x"],
                                          df[left_shoulder_idx + "y"] - df[right_shoulder_idx + "y"]) * 180 / np.pi
    df_angles["head_twist_angle"] = np.arctan2(df[left_ear_idx + "x"] - df[right_ear_idx + "x"],
                                          df[left_ear_idx + "y"] - df[right_ear_idx + "y"]) * 180 / np.pi
    
    

    ####################
    # Body Lean angles
    ####################
    # get indices for necessary body parts
    nose_idx = "NOSE_"
    left_hip_idx = "LEFT_HIP_"
    right_hip_idx = "RIGHT_HIP_"
    Mid_hip_x = (df[left_hip_idx + "x"]+df[right_hip_idx + "x"])/2
    Mid_hip_z = (df[left_hip_idx + "z"]+df[right_hip_idx + "z"])/2
    
    # calculate overall arm angles
    df_angles["body_angle_lon"] = np.arctan2(df[left_shoulder_idx + "y"] - df[right_shoulder_idx + "y"],
                                          df[left_shoulder_idx + "z"] - df[right_shoulder_idx + "z"]) * 180 / np.pi
    df_angles["body_lean_angle"] = np.arctan2(df[nose_idx + "x"] - Mid_hip_x,
                                          df[nose_idx + "z"] - Mid_hip_z) * 180 / np.pi
    
    
    
    return df_angles

