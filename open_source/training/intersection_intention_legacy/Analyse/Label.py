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
import cv2

# Load the CSV file
file_path = 'TrajectoryDataset\GOPR8000_1.csv'

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

# Add columns for labels
main_df['TrajectoryType'] = "-"
main_df['StartTurn'] = "-"
main_df['EndTurn'] = "-"
main_df['RectTopLeft'] = "-"
main_df['RectBottomRight'] = "-"

class TrajectoryLabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Trajectory Labeling App")
        
        self.current_index = 0
        self.trajectory_data = bicycle_data.reset_index()
        self.start_turn = None
        self.end_turn = None
        self.rect = None
        
        # Create UI elements
        self.create_widgets()
        
        # Load the first trajectory
        self.load_trajectory(self.current_index)
    
    def create_widgets(self):
        self.label = Label(self.root, text="Trajectory Type")
        self.label.grid(row=0, column=0)
        
        self.trajectory_type = StringVar(self.root)
        self.trajectory_type.set("Straight")
        self.trajectory_type_menu = OptionMenu(self.root, self.trajectory_type, "Straight", "Left Turn", "Right Turn")
        self.trajectory_type_menu.grid(row=0, column=1)
        
        self.next_button = Button(self.root, text="Next", command=self.next_trajectory)
        self.next_button.grid(row=0, column=2)
        
        self.prev_button = Button(self.root, text="Previous", command=self.prev_trajectory)
        self.prev_button.grid(row=0, column=3)
        
        self.canvas = Canvas(self.root, width=800, height=600)
        self.canvas.grid(row=1, column=0, columnspan=4)
        
        self.fig, self.ax = plt.subplots()
        # # plt.setp(self.ax, xticks = np.arange(0, 2000, 100), yticks = np.arange(-1000, 0, 100))
        # self.ax.set_xticks(np.arange(0, 2000, 100))
        # self.ax.set_yticks(np.arange(0, 1000, 100))
        # self.ax.invert_yaxis()  # Invert the y-axis to have 1000 at the bottom and 0 at the top
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=4)
        
        self.start_turn_entry = Entry(self.root)
        self.start_turn_entry.grid(row=2, column=0)
        self.end_turn_entry = Entry(self.root)
        self.end_turn_entry.grid(row=2, column=1)
        
        self.label_start = Label(self.root, text="Start Turn Index")
        self.label_start.grid(row=3, column=0)
        self.label_end = Label(self.root, text="End Turn Index")
        self.label_end.grid(row=3, column=1)
        
        self.save_button = Button(self.root, text="Save", command=self.save_labels)
        self.save_button.grid(row=2, column=2)
        
        self.selector = RectangleSelector(self.ax, self.on_select, useblit=True, button=[1],
                                          minspanx=5, minspany=5, spancoords='pixels', interactive=True)
        
        self.canvas.mpl_connect("button_press_event", self.on_click)

        # # Load background image
        # self.background_image = cv2.cvtColor(cv2.imread('TrajectoryDataset\\GOPR8000_1.jpg'), cv2.COLOR_BGR2RGB)
        # self.ax.imshow(self.background_image, extent=[0, self.background_image.shape[1], 1000, 0], alpha=0.5, zorder=-1)

        # Load background image
        self.background_image = cv2.cvtColor(cv2.imread('TrajectoryDataset\\GOPR8000_1.jpg'), cv2.COLOR_BGR2RGB)
        self.ax.imshow(self.background_image, origin='upper', alpha=0.5, zorder=-1)
        self.ax.set_xlim(0, self.background_image.shape[1])
        self.ax.set_ylim(self.background_image.shape[0], 0)
    
    def load_trajectory(self, index):
        self.ax.clear()
        self.ax.imshow(self.background_image, origin='upper', alpha=0.5, zorder=-1)
        self.ax.set_xlim(0, self.background_image.shape[1])
        self.ax.set_ylim(self.background_image.shape[0], 0)
        self.start_turn = None
        self.end_turn = None
        self.rect = None
        
        traj = self.trajectory_data.at[index, 'Trajectory']
        x, y = traj['x[px]'], traj['y[px]']
        self.ax.plot(traj['x[px]'], traj['y[px]'], marker="o", markevery=[0], markeredgecolor="green", label=f'Track {self.trajectory_data.at[index, "TrackID"]},green start,red end')
        self.ax.plot(x, y, marker="o", markevery=[-1], markeredgecolor="red")
        self.ax.legend()
        self.canvas.draw()
    
    def next_trajectory(self):
        self.current_index = (self.current_index + 1) % len(self.trajectory_data)
        self.load_trajectory(self.current_index)
    
    def prev_trajectory(self):
        self.current_index = (self.current_index - 1) % len(self.trajectory_data)
        self.load_trajectory(self.current_index)
    
    def find_nearest_point(self, x, y, traj):
        distances = np.sqrt((traj['x[px]'] - x) ** 2 + (traj['y[px]'] - y) ** 2)
        nearest_index = distances.idxmin()
        return traj.at[nearest_index, 'x[px]'], traj.at[nearest_index, 'y[px]'], nearest_index
    
    def on_click(self, event):
        if event.inaxes is not None:
            traj = self.trajectory_data.at[self.current_index, 'Trajectory']
            if self.start_turn is None:
                x, y, idx = self.find_nearest_point(event.xdata, event.ydata, traj)
                self.start_turn = [str(x), str(y)]
                self.ax.plot(x, y, 'go')  # Green for start
            elif self.end_turn is None:
                x, y, idx = self.find_nearest_point(event.xdata, event.ydata, traj)
                self.end_turn = [str(x), str(y)]
                self.ax.plot(x, y, 'ro')  # Red for end
                self.draw_rectangle()
            self.canvas.draw()
    
    def draw_rectangle(self):
        if self.rect is not None:
            self.rect.remove()
        x1, y1 = map(float, self.start_turn)
        x2, y2 = map(float, self.end_turn)
        lower_left_x = min(x1, x2)
        lower_left_y = min(y1, y2)
        width = np.abs(x2 - x1)
        height = np.abs(y2 - y1)
        
        self.rect = Rectangle((lower_left_x, lower_left_y), width, height, linewidth=1, edgecolor='b', facecolor='none')
        self.ax.add_patch(self.rect)
        self.canvas.draw()
    
    def on_select(self, eclick, erelease):
        if self.rect is None:
            self.rect = Rectangle((eclick.xdata, eclick.ydata), erelease.xdata - eclick.xdata, erelease.ydata - eclick.ydata, linewidth=1, edgecolor='b', facecolor='none')
            self.ax.add_patch(self.rect)
        else:
            self.rect.set_xy((min(eclick.xdata, erelease.xdata), min(eclick.ydata, erelease.ydata)))
            self.rect.set_width(np.abs(eclick.xdata - erelease.xdata))
            self.rect.set_height(np.abs(eclick.ydata - erelease.ydata))
        self.adjust_rectangle()
        self.canvas.draw()
    
    def adjust_rectangle(self):
        if self.start_turn and self.end_turn:
            x1, y1 = map(float, self.start_turn)
            x2, y2 = map(float, self.end_turn)
            lower_left_x = min(x1, x2)
            lower_left_y = min(y1, y2)
            width = np.abs(x2 - x1)
            height = np.abs(y2 - y1)
            
            self.rect.set_xy((lower_left_x, lower_left_y))
            self.rect.set_width(width)
            self.rect.set_height(height)
    
    def save_labels(self):
        traj_type = self.trajectory_type.get()
        start_turn = ','.join(self.start_turn) if traj_type != "Straight" else "-"
        end_turn = ','.join(self.end_turn) if traj_type != "Straight" else "-"
        rect_bottom_left = str(self.rect.get_x()) + "," + str(self.rect.get_y()) if traj_type != "Straight" else "-"
        rect_top_right = str(self.rect.get_x() + self.rect.get_width()) + "," + str(self.rect.get_y() + self.rect.get_height()) if traj_type != "Straight" else "-"
        
        self.trajectory_data.at[self.current_index, 'TrajectoryType'] = traj_type
        self.trajectory_data.at[self.current_index, 'StartTurn'] = start_turn
        self.trajectory_data.at[self.current_index, 'EndTurn'] = end_turn
        self.trajectory_data.at[self.current_index, 'RectBottomLeft'] = rect_bottom_left
        self.trajectory_data.at[self.current_index, 'RectTopRight'] = rect_top_right
        
        print(f"Saved: {traj_type}, {start_turn}, {end_turn}, {rect_bottom_left}, {rect_top_right}")
        print(self.trajectory_data.loc[self.current_index])
        
        # Reset for the next trajectory
        self.start_turn = None
        self.end_turn = None
        if self.rect is not None:
            self.rect.remove()
            self.rect = None
        self.canvas.draw()



def main():
    global COVER_PATH
    COVER_PATH = 'TrajectoryDataset\\GOPR8000_1.jpg'
    root = Tk()
    app = TrajectoryLabelingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()