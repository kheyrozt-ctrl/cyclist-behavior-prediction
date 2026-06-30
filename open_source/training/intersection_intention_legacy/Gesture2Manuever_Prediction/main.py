import cv2
import mediapipe as mp
from datetime import datetime
from input import initialize_camera, initialize_video
from output import prepare_output_files, log_keypoints_to_csv, write_frame_to_video, release_video_writer, create_output_directory, initialize_csv_file
from detection import detect_movement
import sys
import os
import argparse
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description='Skeleton and Movement Detection')
    parser.add_argument('--video', type=str, help='Path to the video file. If not provided, the camera will be used.')
    parser.add_argument('--camera', type=int, default=0, help='Camera index to use (default: 0).')
    parser.add_argument('--record_csv', action='store_true', help='Record skeleton data to a CSV file.')
    parser.add_argument('--record_video_plain', action='store_true', help='Record plain video data without skeletons or movements.')
    parser.add_argument('--record_video_with_skeleton', action='store_true', help='Record video data with skeletons but without movements.')
    parser.add_argument('--record_video_with_skeleton_and_movement', action='store_true', help='Record video data with both skeletons and movements.')
    return parser.parse_args()

def main(args=None):
    if args is None:
        args = parse_arguments()

    # Initialize MediaPipe Pose solution
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.8, min_tracking_confidence=0.5)
    

    # Initialize the camera or video
    if args.video:
        cap = initialize_video(args.video)
    else:
        cap = initialize_camera(cam=args.camera)

    # Check if the camera or video is opened properly
    if not cap.isOpened():
        print("Error: Could not open camera or video.")
        return

    # Get frame dimensions and ensure they are valid
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width == 0 or frame_height == 0:
        print("Error: Camera or video frame width or height is zero.")
        return

    print(f"Frame dimensions: {frame_width}x{frame_height}")

    # Prepare output files based on the provided arguments
    output_dir = 'Output_Data'
    create_output_directory(output_dir)

    csv_file = None
    video_writer_plain = None
    video_writer_skeleton = None
    video_writer_skeleton_and_movement = None

    if args.record_csv:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.video:
            total_frames_num = int (cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            video_filename = os.path.splitext(os.path.basename(args.video))[0]
            csv_file = f'{output_dir}/video{video_filename}_{timestamp}_skeleton.csv'
        else:
            csv_file = f'{output_dir}/{timestamp}_skeleton.csv'
        initialize_csv_file(csv_file)

    if args.record_video_plain:
        _, video_writer_plain = prepare_output_files(output_dir, frame_width, frame_height)

    if args.record_video_with_skeleton:
        _, video_writer_skeleton = prepare_output_files(output_dir, frame_width, frame_height)

    if args.record_video_with_skeleton_and_movement:
        _, video_writer_skeleton_and_movement = prepare_output_files(output_dir, frame_width, frame_height)

    retry_count = 5
    retry_delay = 1  # seconds

    while retry_count > 0:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera/video. Retrying...")
            retry_count -= 1
            time.sleep(retry_delay)
        else:
            break

    if retry_count == 0:
        print("Error: Could not read frame from camera/video after multiple attempts.")
        return

    framenum = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("For video: End of video; For camera: Error: Could not read frame from camera.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            if args.record_csv:
                # Log keypoints to CSV without movement prediction
                if args.video:
                    framenum = framenum + 1
                    timestamp = framenum/video_fps
                    timestamp = float(f"{timestamp:.3f}")
                else:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                log_keypoints_to_csv(landmarks, timestamp, csv_file)

            # Detect movement
            movement = detect_movement(landmarks, mp_pose)
            
            # Annotate display frame with skeleton and movement predictions
            display_frame = frame.copy()
            mp.solutions.drawing_utils.draw_landmarks(
                display_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_pose_landmarks_style())
            cv2.putText(display_frame, f"Movement: {movement}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            if args.record_video_plain:
                # Write the plain frame to the video without skeletons or movements
                write_frame_to_video(video_writer_plain, frame)

            if args.record_video_with_skeleton:
                # Create a copy of the frame with skeletons but without movement predictions for recording
                frame_for_recording_skeleton = frame.copy()
                mp.solutions.drawing_utils.draw_landmarks(
                    frame_for_recording_skeleton, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_pose_landmarks_style())
                # Write the frame to the video
                write_frame_to_video(video_writer_skeleton, frame_for_recording_skeleton)

            if args.record_video_with_skeleton_and_movement:
                # Create a copy of the frame with skeletons and movement predictions for recording
                frame_for_recording_skeleton_and_movement = frame.copy()
                mp.solutions.drawing_utils.draw_landmarks(
                    frame_for_recording_skeleton_and_movement, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_pose_landmarks_style())
                cv2.putText(frame_for_recording_skeleton_and_movement, f"Movement: {movement}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                # Write the frame to the video
                write_frame_to_video(video_writer_skeleton_and_movement, frame_for_recording_skeleton_and_movement)
        
        # cv2.imshow('Movement and Skeleton Detection', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if args.record_video_plain:
        # Release plain video writer
        release_video_writer(video_writer_plain)

    if args.record_video_with_skeleton:
        # Release skeleton video writer
        release_video_writer(video_writer_skeleton)

    if args.record_video_with_skeleton_and_movement:
        # Release skeleton and movement video writer
        release_video_writer(video_writer_skeleton_and_movement)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
