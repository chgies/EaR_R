import cv2
import mediapipe as mp
import csv
import threading
import datetime
import json
import concurrent.futures

# Global variables definition
video_parts_to_analyze = []
thread_local = threading.local()
current_output_path = ""
current_video_path = ""
# Filepath od extracted dataset
dataset_filepath = '/media/christoph/Crucial X8/Abschlussarbeit_Datasets/CANDOR/processed_dataset/processed/00ae2f18-9599-4df6-8e3a-6936c86b97f0'
# user ids of dataset, used for mp4 filename
video_user_ids = json.load(open(dataset_filepath + '/processed/channel_map.json'))
print(video_user_ids)
video_paths = [dataset_filepath + '/processed/' + video_user_ids.get('L') + '.mp4']#, filepath + '/processed/' + user_names.get('R') + '.mp4']
print(video_paths)
output_csv_paths = [dataset_filepath + '/processed/mp_data/' + video_user_ids.get('L') + '.csv']#, filepath + '/processed/mp_data/' + user_names.get('R') + '.csv']

# Save the CSV data to a file
def write_to_file(in_output_path, in_csv_data):
    with open(in_output_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if csv.reader(csvfile).line_num == 0:
            csv_writer.writerow(['frame_number', 'landmark', 'x', 'y', 'z'])
        csv_writer.writerows(in_csv_data)

# Thia function converts the list of landmarks found by mediapipe into csv data
# as preparation for writing to a .csv file       
def convert_landmarks_to_csv(in_landmarks, in_frame_number, out_csv_data):
    for idx, landmark in enumerate(in_landmarks):
        out_csv_data.append([in_frame_number, mp_pose.PoseLandmark(idx).name, landmark.x, landmark.y, landmark.z])

# This function analyzes the video part defined by its start and end frame,
# grabs the pose data out of each frame and calls the write_to_file function to write
# data into a csv file
def analyze_frames(in_video_part):
    global current_output_path
    global current_video_path
    cv2_screen_capture = cv2.VideoCapture(current_video_path)
    
    #start at start_frame of current video part
    cv2_screen_capture.set(cv2.CAP_PROP_POS_FRAMES, in_video_part[0])
    local_csv_data = []
    frame_index = in_video_part[0]
    
    # Used for information purposes
    frames_to_process = in_video_part[1] - in_video_part[0]
    
    while cv2_screen_capture.isOpened() and cv2_screen_capture.get(cv2.CAP_PROP_POS_FRAMES) <= in_video_part[1]:
        ret, current_frame = cv2_screen_capture.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_as_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        processed_frame = pose.process(frame_as_rgb)

        # Get the pose landmarks of the frame
        if processed_frame.pose_landmarks:
            # mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Add the landmark coordinates to the local_csv_data list
            convert_landmarks_to_csv(processed_frame.pose_landmarks.landmark, frame_index, local_csv_data)

        frame_index += 1
        frames_to_process -= 1
        print(f"{frames_to_process} Frames still to be analyzed in this process")
    
    cv2_screen_capture.release()
    cv2_screen_capture.close()
    
    print(f"Landmarks found in this task: {local_csv_data}")
    
    # write found data into csv file
    write_to_file(current_output_path, local_csv_data)
    
    print("Task done")

# Manage Threading, gets called by Main function and calls analyzing function
def get_poses(in_vidoe_parts):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(analyze_frames, in_vidoe_parts)


# Main function
if __name__ == "__main__":

    # For testing purposes: measure program runtime
    start_time = datetime.datetime.now()

    # Initialise MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()
    
    for video in video_paths:    
        path_index = video_paths.index(video)
        current_output_path = output_csv_paths[path_index]
        current_video_path = video_paths[path_index]
        total_frames = 200 #Original: int(cv2.VideoCapture(video_paths[path_index]).get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Divice Video into smaller parts for multithreading
        number_of_parts = 4
        frames_per_part = total_frames // number_of_parts
        video_parts_to_analyze = []
        for i in range(number_of_parts):            
            current_start_frame = i * frames_per_part
            current_end_frame = (i + 1) * frames_per_part - 1 if i != number_of_parts - 1 else total_frames - 1
            current_video_part = [current_start_frame, current_end_frame]
            video_parts_to_analyze.append(current_video_part)
        
        # Start analyzing
        get_poses(video_parts_to_analyze) 
            
        cv2.destroyAllWindows()
    
    # For testing purposes
    end_time = datetime.datetime.now()
    print(f"Aufgabe dauerte {end_time-start_time}")    