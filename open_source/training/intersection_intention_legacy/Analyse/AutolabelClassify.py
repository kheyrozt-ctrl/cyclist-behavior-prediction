import numpy as np
from matplotlib.path import Path
import ExtractTrajectory
import os

# Load the saved mask data
# mask_data = np.load("Results\\mask_data.npy", allow_pickle=True).item()

def point_in_polygon(point, polygon):
    path = Path(polygon)
    return path.contains_point(point)

def classify_trajectory(start_point, end_point,mask_data):
    start_mask = None
    end_mask = None

    for mask_label, verts in mask_data.items():
        if point_in_polygon(start_point, verts):
            start_mask = mask_label
        if point_in_polygon(end_point, verts):
            end_mask = mask_label

    if start_mask and end_mask:
        if (start_mask == "North" and end_mask == "South") or (start_mask == "South" and end_mask == "North") or (start_mask == "West" and end_mask == "East") or (start_mask == "East" and end_mask == "West"):
            return "Straight"
        elif (start_mask == "North" and end_mask == "East") or (start_mask == "East" and end_mask == "South") or (start_mask == "South" and end_mask == "West") or (start_mask == "West" and end_mask == "North"):
            return "Left Turn"
        elif (start_mask == "East" and end_mask == "North") or (start_mask == "South" and end_mask == "East") or (start_mask == "West" and end_mask == "South") or (start_mask == "North" and end_mask == "West"):
            return "Right Turn"
        else:
            return "Unknown"
    else:
        return "Unknown"

# # Example trajectory points
# start_point = (100, 150)  # Replace with actual start point
# end_point = (200, 250)    # Replace with actual end point


def ClassifyALLTrajectory(file_path,mask_filepath):
    mask_data = np.load(mask_filepath, allow_pickle=True).item()
    bicycle_data = ExtractTrajectory.ExtractCyclistTrajectory(file_path)
    for index, row in bicycle_data.iterrows():
        trajectory = row['Trajectory']
        
        start_point = (trajectory['x[px]'][4],trajectory['y[px]'][4])
        end_point = (trajectory.iloc[-4,0],trajectory.iloc[-4,1])   

        trajectory_type = classify_trajectory(start_point, end_point,mask_data)

        # row['TrajectoryType'] = trajectory_type
        bicycle_data.at[index, 'TrajectoryType'] = trajectory_type
        print(f"The trajectory is classified as: {trajectory_type}")

    last_slash_index = file_path.rfind("\\")
    csvfilename = "Results\\LabeledTrajDataset\\" + file_path[last_slash_index+1:]
    bicycle_data.to_csv(csvfilename)                              
        
