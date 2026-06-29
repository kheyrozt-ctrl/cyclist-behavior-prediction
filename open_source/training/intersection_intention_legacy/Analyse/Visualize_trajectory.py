import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import csv
import re

# Load the CSV file
file_path = 'TrajectoryDataset\GOPR8000_1.csv'

# Define critical distance and angle for turning definition
critical_distance = 100  # in pixels
turn_threshold = np.pi / 8  # 22.5 degrees






# Read the CSV file with the specified delimiter and handle line endings
with open(file_path, 'r',  newline='\r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile.read().splitlines(), delimiter=';')
    data_list = list(reader)

# Set the correct column names
columns = 'TrackID,Type,EntryGate,EntryTime[ms],ExitGate,ExitTime[ms],TraveledDist[px],AvgSpeed[Kpx/h],Trajectory'.split(',')
main_data = []
# data = pd.DataFrame(data_list[1:], columns=columns)

# Process each row to separate main data and trajectory data
for row in data_list[1:]:
    main_row = row[:8]  # First 8 elements
    
    trajectory_data = row[8:]  # Remaining elements
    trajectory_df = pd.DataFrame(
        [trajectory_data[i:i+5] for i in range(0, len(trajectory_data)-4, 5)],
        columns=['x[px]', 'y[px]', 'Speed[Kpx/h]', 'TotalAccel[pxs-2]', 'Time[ms]']
    )
    trajectory_df[['x[px]', 'y[px]', 'Speed[Kpx/h]', 'TotalAccel[pxs-2]', 'Time[ms]']] = trajectory_df[['x[px]', 'y[px]', 'Speed[Kpx/h]', 'TotalAccel[pxs-2]', 'Time[ms]']].astype('float32')
    main_row.append(trajectory_df)
    main_data.append(main_row)

# Create the main DataFrame
main_df = pd.DataFrame(main_data, columns=columns)

main_df[['TraveledDist[px]', 'AvgSpeed[Kpx/h]']] = main_df[['TraveledDist[px]', 'AvgSpeed[Kpx/h]']].astype('float32')


# Filter the data for Bicycle type
bicycle_data = main_df[main_df['Type'] == ' Bicycle']



# Function to calculate direction change in bicycle's local coordinate system
def calculate_turns(traj, critical_distance):
    traj['dist'] = np.sqrt(traj['x[px]'].diff()**2 + traj['y[px]'].diff()**2).cumsum().fillna(0)
    traj['angle'] = np.arctan2(traj['y[px]'].diff(), traj['x[px]'].diff()).fillna(0)
    traj['angle_diff'] = traj['angle'].diff().fillna(0)
    
    turn_segments = []
    current_segment = {'start': 0, 'end': None, 'type': 'straight'}
    count = 1
    # for i in range(1, len(traj)):
    while count< len(traj):
        i = count
        if traj.loc[i, 'dist'] - traj.loc[current_segment['start'], 'dist'] >= critical_distance or i == len(traj) - 1:
            segment_angle_diff = traj.loc[current_segment['start']:i, 'angle_diff'].sum()
            if segment_angle_diff > turn_threshold:
                current_segment['type'] = 'left'
            elif segment_angle_diff < -turn_threshold:
                current_segment['type'] = 'right'
            else:
                current_segment['type'] = 'straight'
            current_segment['end'] = i
            turn_segments.append(current_segment.copy())
            count = current_segment['start'] + 1
            current_segment = {'start': count, 'end': None, 'type': 'straight'}
            
        count += 1


    
    return turn_segments



# Function to calculate forward direction
def calculate_forward_direction(traj):
    traj['forward_x'] = traj['x[px]'].diff().fillna(0)
    traj['forward_y'] = traj['y[px]'].diff().fillna(0)
    traj['forward_norm'] = np.sqrt(traj['forward_x']**2 + traj['forward_y']**2)
    traj['forward_x'] /= traj['forward_norm']
    traj['forward_y'] /= traj['forward_norm']
    return traj

# Apply the direction calculation to each trajectory
for index, row in bicycle_data.iterrows():
    traj = row['Trajectory']
    traj = calculate_forward_direction(traj)
    turn_segments = calculate_turns(traj, critical_distance)
    # print(f'Track ID: {row["TrackID"]}, Turn Segments: {turn_segments}')



# fig = plt.figure(figsize=(10, 6))
# count = 0
# for index, row in bicycle_data.iterrows():
#     if count < 10:
#         traj = row['Trajectory']
#         print(f'Track ID: {row["TrackID"]}, Turn Segments: {turn_segments}')
#         plt.plot(traj['x[px]'], traj['y[px]'], marker="o", label=f'Track {row["TrackID"]}')
        
#     count += 1

fig = plt.figure(figsize=(10, 6))
for index, row in bicycle_data.iterrows():

    traj = row['Trajectory']
    # print(f'Track ID: {row["TrackID"]}, Turn Segments: {turn_segments}')
    # plt.plot(traj['x[px]'], traj['y[px]'], marker="o", label=f'Track {row["TrackID"]}')
    plt.plot(traj['x[px]'], -1*traj['y[px]'], label=f'Track {row["TrackID"]}')
    
   
    
    
    
my_x_ticks = np.arange(0, 2000, 100)
my_y_ticks = np.arange(-1000, 0, 100)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.xlabel('X [px]')
plt.ylabel('Y [px]')
plt.title('Bicycle Trajectories')
plt.legend()
plt.show()