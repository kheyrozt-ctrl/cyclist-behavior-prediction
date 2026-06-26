import AutolabelMask
import AutolabelScreenshot
import AutolabelClassify
from tkinter import Tk
import sys
import os


# def on_masks_saved():
    # # Load the CSV file
    # file_path = 'TrajectoryDataset\\GOPR8000_1.csv'
    # print(file_path)
    # AutolabelClassify.ClassifyALLTrajectory(file_path)

#     # Exit the program
#     sys.exit()

# if __name__ == "__main__":
#     VIDEO_PATH = 'TrajectoryDataset\\GOPR8000_1.mp4'  # Video path
#     COVER_PATH = "TrajectoryDataset\\GOPR8000_1.jpg"  # Image path
#     AutolabelScreenshot.IntersectionVideoCapture(VIDEO_PATH, COVER_PATH)
    
#     root = Tk()
#     app = AutolabelMask.MaskLabelingApp(root, on_masks_saved)
#     root.mainloop()

def on_masks_saved(file_path,mask_filepath):
    print(file_path)
    AutolabelClassify.ClassifyALLTrajectory(file_path,mask_filepath)

    
    # 继续下一个文件的处理
    process_next_file()

def process_file(video_path):
    
    cover_path = video_path.replace('.MP4', '.jpg')
    file_path = video_path.replace('.MP4', '.csv')
    mask_filename = os.path.splitext(os.path.basename(video_path))[0]  # 获取文件名，不含扩展名
    mask_filepath = os.path.join("Results", "Masks", f"{mask_filename}.npy")
    os.makedirs(os.path.dirname(mask_filepath), exist_ok=True)  # 确保目录存在
    
    AutolabelScreenshot.IntersectionVideoCapture(video_path, cover_path)
    
    root = Tk()
    app = AutolabelMask.MaskLabelingApp(root, mask_filepath, lambda: on_masks_saved(file_path,mask_filepath),cover_path)
    root.mainloop()

def process_next_file():
    
    global files_to_process
    
    if files_to_process:
        
        video_path = files_to_process.pop(0)
        
        process_file(video_path)
        
    else:
        # 所有文件处理完毕，退出程序
        sys.exit()

if __name__ == "__main__":
    folder_path = 'TrajectoryDataset'
    files_to_process = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.MP4')]
    print(files_to_process)
    process_next_file()

