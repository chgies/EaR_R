import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import time
import csv 

from datamanager_candor import get_biggest_files
#from datamanager_emoreact import create_dataframe, get_df_test, get_df_train, get_df_val

output_window = None    # Used as screen variable to draw landmarks on
last_timestamp_ms = 0   # needed by Mediapipe when in "LIVE_STREAM"-RunningMode
frame = 0               # The current frame of the video, used to write pose data to csv
csv_write_count = 0
csv_write_mean = 0.0
detect_both_count = 0
detect_both_mean = 0.0
draw_landmarks_count = 0
draw_landmarks_mean = 0.0
print_result_count = 0
print_result_mean = 0.0
analyze_video_count = 0
analyze_video_mean = 0.0

def measure_performance(timestamp, last_result, cycle_count):
    """
    This function measures the mean duration of other functions in this file.
    Gets called at each running cycle of a function

    Parameters:
        timestamp (time.time): the starting timestamp of the current function call
        last_result: the mean duration time up of the cycle before
        cycle_count: how many times the function has been called until now

    Result:
        mean_duration: the newest mean duration time of the function
    """
    now = time.time()
    duration = now - timestamp
    mean_duration = (((cycle_count-1)*last_result)+duration)/cycle_count
    return mean_duration


def write_pose_to_csv(path, frame_number, id, landmarks):
    """
    Takes landmark data and writes it to a csv file for further processing.
    Gets called in draw_landmarks().

    Parameters:
    path (string): The path of the video
    frame_number (integer): the global variable of the current frame
    id (mediapipe.framework.formats.landmark_pb2): the id of the pose found in the current frame. One id for each person
    landmarks(mediapipe.framework.formats.landmark_pb2): the landmarks of the found id

    ------

    The landmark has the following structure, with each value in float format (x,y,z: [-1,1], visibility, presence: [0,1]):
    [NormalizedLandmark(x, y, z, visibility, presence), NormalizedLandmark(x, y, z, visibility, presence)...]
    
    """
    global csv_write_count
    global csv_write_mean

    csv_start_time = time.time()
    csv_write_count  += 1

    output_dir = path + '/pose.csv'
    converted_csv_data = []
    for landmark in (landmarks):
        converted_csv_data.append([frame_number, id, landmark.x, landmark.y, landmark.z])

    with open(output_dir, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if csv.reader(csvfile).line_num == 0:
            csv_writer.writerow(['frame_number', 'id', 'x', 'y', 'z'])
        csv_writer.writerows(converted_csv_data)

    csv_write_mean = measure_performance(csv_start_time, csv_write_mean, csv_write_count)
    print(f"write_to_csv mean duration: {csv_write_mean}")

def detect_both_online(image):
    """
    Detects whether both screens are online based on the given image. Only need if there are 2 screens in the video.

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    bool: True if both screens are online, False otherwise.
    """
    global detect_both_mean 
    global detect_both_count 
    detect_both_start_time = time.time()
    detect_both_count += 1

    left_online = False
    right_online = False
    width = image.shape[1]
    left = image[:, :width//2]
    right = image[:, width//2:]
    mean_left = np.mean(left)
    #print(mean_left)
    if mean_left > 10:
        left_online = True
    if np.mean(right) > 10:
        right_online = True

    detect_both_mean = measure_performance(detect_both_start_time, detect_both_mean, detect_both_count)
    print(f"write_to_csv mean duration: {detect_both_mean}")
    
    if left_online and right_online:
        return True
    else:
        return False

def draw_landmarks(image, results):
    """
    Draws landmarks on the given image using the results from pose estimation.

    Args:
        image (numpy.ndarray): The input image.
        results (mediapipe.python.solution_base.SolutionOutputs): The pose estimation results.

    Returns:
        numpy.ndarray: The image with landmarks drawn.
    """
    
    global draw_landmarks_mean 
    global draw_landmarks_count
    draw_landmarks_start_time = time.time()
    draw_landmarks_count += 1


    pose_landmarks_list = results.pose_landmarks
    annotated_image = np.copy(image)
    global frame

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        #print(f"IDX: {idx}, LM: {pose_landmarks}")
        write_pose_to_csv('/media/christoph/Crucial X8/Abschlussarbeit_Datasets/CANDOR/processed_dataset/processed/fffda3e6-7d99-4db8-aa12-16e99fa454c2/processed',idx, frame,pose_landmarks)

        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()

        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z) for landmark in pose_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style())
        
    draw_landmarks_mean = measure_performance(draw_landmarks_start_time, draw_landmarks_mean, draw_landmarks_count)
    print(f"draw_landmarks mean duration: {detect_both_mean}")
    return annotated_image

def print_result(detection_result: vision.PoseLandmarkerResult, output_image: mp.Image,
                 timestamp_ms: int):
    """
    Gets called automatically by Mediapipe Pose Landmarker. Draws found landmarks to 
    given image and returns it to global variable output_window

    Args:
        image (numpy.ndarray): The input image.
        results: the output window has landmarks drawn on it.

    """

    global print_result_mean 
    global print_result_count 
    print_result_start_time = time.time()
    print_result_count += 1

    global output_window
    global last_timestamp_ms

    if timestamp_ms < last_timestamp_ms:
        return
    last_timestamp_ms = timestamp_ms
    # print("pose landmarker result: {}".format(detection_result))
    output_window = cv2.cvtColor(
        draw_landmarks(output_image.numpy_view(), detection_result), cv2.COLOR_RGB2BGR)
    
    print_result_mean = measure_performance(print_result_start_time, print_result_mean, print_result_count)
    print(f"print_results mean duration: {print_result_mean}")


def analyze_video(path):
    """
    Analyzes a video file or webcam stream and detects human poses using MediaPipe Pose.

    Args:
        path (str): The path to the video file or '0' for webcam stream.
    Returns:
        None
    """

    global analyze_video_mean 
    global analyze_video_count

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
    
        analyze_video_start_time = time.time()
        analyze_video_count += 1
        
        cap = cv2.VideoCapture(path)
        global frame
        frame = 0
        while cap.isOpened():
            success, image = cap.read()
            start_time = time.time()
            #print(f"Frame: {frame}")
            if not success:
                break

            if not detect_both_online(image):
                continue

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)            
            
            if output_window is not None:
                cv2.imshow("MediaPipe Pose Landmark", output_window)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            end_time = time.time()
            print(f"FPS: {1.0 / (end_time-start_time)}")
            frame += 1

            analyze_video_mean = measure_performance(analyze_video_start_time, analyze_video_mean, analyze_video_count)
            print(f"write_to_csv mean duration: {analyze_video_mean}")

def test_emoreact():
    df_test = get_df_test()
    df_train = get_df_train()
    df_val = get_df_val()
    print(df_test)
    print(df_train)
    print(df_val)
    for index, row in df_test.iterrows():
        analyze_video(row['video'])
        print("done")

def test_candor():
    files = get_biggest_files()
    print(files)
    for file in files:
        analyze_video(file)
        print("done")


if __name__ == "__main__":

    #test_emoreact()

    ### Basic Options used to initialize Mediapipe Pose Detector

    # landmark files available on https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
    base_options = python.BaseOptions(model_asset_path='./landmark_files/pose_landmarker_heavy.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_poses=2,
        min_pose_detection_confidence=0.7,
        min_pose_presence_confidence=0.7,
        min_tracking_confidence=0.8,
        output_segmentation_masks=False,
        result_callback=print_result
    )
    test_candor()
