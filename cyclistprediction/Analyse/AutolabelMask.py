import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk, Button, filedialog
from matplotlib.patches import Polygon
from matplotlib.widgets import PolygonSelector
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os




class MaskLabelingApp:
    def __init__(self, root, mask_filepath, callback,cover_path):
        self.root = root
        self.callback = callback
        self.mask_filepath = mask_filepath  # 添加文件名参数
        self.cover_path = cover_path
        self.root.title("Mask Labeling App")

        self.save_button = Button(root, text="Save Masks", command=self.save_masks)
        self.save_button.pack()

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()

        self.image = None
        self.polygon_selector = None

        self.mask_labels = ["North", "East", "South", "West"]
        self.mask_data = {}
        self.current_polygon_index = 0

        self.load_image()

    def load_image(self):
        # COVER_PATH = "TrajectoryDataset\\GOPR8000_1.jpg"  # Image path 
        self.image = cv2.cvtColor(cv2.imread(self.cover_path), cv2.COLOR_BGR2RGB)
        

        self.ax.clear()
        self.ax.imshow(self.image)
        self.canvas.draw()

        self.start_polygon_selection()

    def start_polygon_selection(self):
        if self.current_polygon_index < len(self.mask_labels):
            self.current_polygon = self.mask_labels[self.current_polygon_index]
            self.polygon_selector = PolygonSelector(self.ax, self.onselect, useblit=True)

    def onselect(self, verts):
        if self.current_polygon:
            adjusted_verts = [(0 if abs(x) <= 50 else x, 0 if abs(y) <= 50 else y) for x, y in verts]
            polygon = Polygon(adjusted_verts, closed=True, edgecolor='r', facecolor='none', lw=2)
            self.ax.add_patch(polygon)
            self.mask_data[self.current_polygon] = adjusted_verts
            self.canvas.draw()

            self.current_polygon_index += 1
            if self.current_polygon_index < len(self.mask_labels):
                self.start_polygon_selection()
            else:
                self.polygon_selector.disconnect_events()
                self.current_polygon = None

    def save_masks(self):
        for label, verts in self.mask_data.items():
            print(f"Mask for {label}: {verts}")

        # Save the mask data to a file
        # np.save("Results\\Masks\\mask_data.npy", self.mask_data)
        np.save(self.mask_filepath, self.mask_data)
        print("Masks saved to {mask_save_path}")
        
         # Call the callback function to continue execution
        if self.callback:
            self.callback()

        # Automatically close the application after saving
        self.root.destroy()



