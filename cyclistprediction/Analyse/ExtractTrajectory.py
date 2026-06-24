import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import csv
import re
from tkinter import Tk, Label, Button, Canvas, Entry, StringVar, OptionMenu, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector


def ExtractCyclistTrajectory(file_path):
    

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

    # Add columns for labels
    main_df['TrajectoryType'] = "-"

    # Filter the data for Bicycle type
    bicycle_data = main_df[main_df['Type'] == ' Bicycle']

    

    return bicycle_data