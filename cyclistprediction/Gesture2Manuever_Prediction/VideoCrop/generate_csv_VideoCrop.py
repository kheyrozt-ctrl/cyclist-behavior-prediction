import csv
import os
import argparse

def generate_csv(directory, csv_path):
    """Generate a CSV file with video file names for cropping times."""
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Video File', 'Start Time (sec)', 'End Time (sec)'])
        for filename in os.listdir(directory):
            if filename.endswith('.mp4'):
                writer.writerow([filename, '', ''])  # Leave times blank for user input

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate CSV for video cropping times')
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing videos')
    parser.add_argument('--csv_path', type=str, required=True, help='Output path for the CSV file')
    args = parser.parse_args()

    generate_csv(args.video_dir, args.csv_path)
    print(f"Generated CSV file at {args.csv_path}. Please fill in the start and end times for each video.")
