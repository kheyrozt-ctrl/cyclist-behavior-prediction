import csv
import os
import argparse
import subprocess

def read_crop_times(csv_path):
    """Read the cropping times from the CSV file."""
    crop_times = {}
    with open(csv_path, newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            video_file = row['Video File'].strip()  # Ensure the file name has no leading/trailing spaces
            start_time = '0'
            end_time = row['End Time (sec)'].strip()

            # Check if the start and end times are not empty
            if start_time and end_time:
                try:
                    start_time = float(start_time)
                    end_time = float(end_time)
                    crop_times[video_file] = (start_time, end_time)
                except ValueError:
                    print(f"Error: Invalid time format in row for video {video_file}. Please ensure times are numeric.")
            else:
                print(f"Warning: Skipping {video_file} due to missing start or end time.")
    
    return crop_times

def crop_video(input_path, output_path, start_time, end_time):
    """Crop the video using FFMPEG."""
    command = [
        'ffmpeg',
        '-i', input_path,
        '-ss', str(start_time),
        '-to', str(end_time),
        '-c', 'copy',
        output_path
    ]
    subprocess.run(command, check=True)

def process_videos(directory, output_directory, crop_times):
    """Process each video based on the crop times provided in the CSV file."""
    for video_file, times in crop_times.items():
        input_path = os.path.join(directory, video_file)
        output_path = os.path.join(output_directory, video_file)
        start_time, end_time = times
        print(f"Cropping {video_file} from {start_time} to {end_time} seconds.")
        crop_video(input_path, output_path, start_time, end_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crop videos based on provided times in a CSV file')
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing videos')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save cropped videos')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file with crop times')
    args = parser.parse_args()

    crop_times = read_crop_times(args.csv_path)
    # print(crop_times)
    process_videos(args.video_dir, args.output_dir, crop_times)