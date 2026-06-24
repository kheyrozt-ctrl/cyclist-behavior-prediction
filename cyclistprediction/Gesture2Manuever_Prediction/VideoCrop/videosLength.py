import os
import cv2

def get_video_duration(video_path):
    """Returns the duration of a video in seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0  # If video can't be opened, return 0
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # Total frames
    cap.release()
    if fps > 0:
        return frame_count / fps
    return 0

def analyze_videos(folder_path):
    """Analyzes all videos in the given folder."""
    video_extensions = (".mp4", ".avi", ".mov", ".mkv")  # Add more formats if needed

    # Initialize counters
    total_videos = 0
    total_duration = 0

    prefix_stats = {
        "1a": {"count": 0, "duration": 0},
        "1b": {"count": 0, "duration": 0},
        "1c": {"count": 0, "duration": 0},
        "1d": {"count": 0, "duration": 0},
        "1e": {"count": 0, "duration": 0},
        "1g": {"count": 0, "duration": 0}
    }

    # Scan the folder for video files
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(video_extensions):  # Check if it's a video
            video_path = os.path.join(folder_path, filename)
            duration = get_video_duration(video_path)

            total_videos += 1
            total_duration += duration

            # Check filename prefix
            for prefix in prefix_stats.keys():
                if filename.startswith(prefix):
                    prefix_stats[prefix]["count"] += 1
                    prefix_stats[prefix]["duration"] += duration

    # Print Results
    print(f"Total Videos: {total_videos}")
    print(f"Total Duration: {total_duration:.2f} seconds\n")

    for prefix, stats in prefix_stats.items():
        print(f"Prefix '{prefix}': {stats['count']} videos, {stats['duration']:.2f} seconds")

# Example usage
folder_path = "GestureDataset/depth_camera_videos_croped"  # Change this to your folder path
analyze_videos(folder_path)
