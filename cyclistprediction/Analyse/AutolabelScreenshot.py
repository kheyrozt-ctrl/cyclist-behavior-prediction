import cv2



def IntersectionVideoCapture(VIDEO_PATH,COVER_PATH):
    # VIDEO_PATH = 'TrajectoryDataset\\GOPR8000_1.mp4'  # Video path
    # COVER_PATH = "TrajectoryDataset\\GOPR8000_1.jpg"  # Image path  
    try:
        vc = cv2.VideoCapture(VIDEO_PATH)  # Open the video
        video_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))  # Video width
        video_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Video height
        vc.set(cv2.CAP_PROP_POS_MSEC, 1000)  # Set the position to 1000 milliseconds
        rval, frame = vc.read()  # Read the frame
        if rval:
            cv2.imwrite(COVER_PATH, frame)  # Save the frame as an image
        else:
            print("Video Capture Write Failed")
    except Exception as e:
        print(f"Video Capture Failed: {e}")