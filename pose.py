import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import time
import csv 
import os
import torch
import tensorflow as tf
from threading import Thread
import concurrent.futures

from datamanager_candor import get_all_movie_files
from datamanager_candor import get_biggest_files
#from datamanager_emoreact import create_dataframe, get_df_test, get_df_train, get_df_val

output_window = None    # Used as screen variable to draw landmarks on
last_timestamp_ms = 0   # needed by Mediapipe when in "LIVE_STREAM"-RunningMode
frame = 0               # The current frame of the video, used to write pose data to csv

analyzed_results = "frame,person,x,y,z\n"

def write_pose_to_csv(person_id):
    global analyzed_results
    output_dir = f'/mnt/g/Abschlussarbeit_Datasets/CANDOR/processed_dataset/processed/fffda3e6-7d99-4db8-aa12-16e99fa454c2/processed'+ f'/pose_{person_id}.csv'
    file = open(output_dir, 'a', newline='')
    file.writelines(analyzed_results)
    analyzed_results = ""
    file.close()
    
def detect_both_online(image):
    """
    Detects whether both screens are online based on the given image. Only need if there are 2 screens in the video.

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    bool: True if both screens are online, False otherwise.
    """

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
    
    pose_landmarks_list = results.pose_landmarks

    annotated_image = np.copy(image)
    global frame

    global analyzed_results
    # draw poses on opencv window.
    for idx in range(len(pose_landmarks_list)):

        pose_landmarks = pose_landmarks_list[idx]
            
        # this is used to save found points for csv files
        for landmark in pose_landmarks:
            new_row = f"{frame},{idx},{landmark.x},{landmark.y},{landmark.z}"
            analyzed_results += ("\n"+new_row)

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

    global output_window
    global last_timestamp_ms

    if timestamp_ms < last_timestamp_ms:
        return
    last_timestamp_ms = timestamp_ms
    
    output_window = cv2.cvtColor(
        draw_landmarks(output_image.numpy_view(), detection_result), cv2.COLOR_RGB2BGR)

    #if RunningMode is VIDEO
    return output_window


def analyze_video(path):
    """
    Analyzes a video file or webcam stream and detects human poses using MediaPipe Pose.

    Args:
        path (str): The path to the video file or '0' for webcam stream.
    Returns:
        None
    """

    if path == '/mnt/g/Abschlussarbeit_Datasets/CANDOR/processed_dataset/processed/fffda3e6-7d99-4db8-aa12-16e99fa454c2/processed/5d5162f1b50a1000169da137.mp4':
        id = 0
    else:
        id = 1
    print(f"Starting analysis of video {id}")

    with vision.PoseLandmarker.create_from_options(options) as landmarker:        
        cap = cv2.VideoCapture(path)
        global frame
        frame = 0
        while cap.isOpened():
            success, image = cap.read()
            start_time = time.time()
            if not success:
                break
            
            if not detect_both_online(image):
                frame += 1
                continue

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            
            # if runningMode is LIVESTREAM
                #landmarker.detect_async(mp_image, timestamp_ms)            
                
            #if RunningMode is VIDEO
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
                      
            output_window = cv2.cvtColor(
                draw_landmarks(image, result), cv2.COLOR_RGB2BGR)
            # end of VIDEO RunningMode code
            
            #if RunningMode is LIVESTREAM
            #if output_window is not None:
                #cv2.imshow("MediaPipe Pose Landmark", output_window)

            #if RunningMode is VIDEO
            if output_window is not None:
                cv2.imshow("MediaPipe Pose Landmark", output_window)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # write found pose data every x frames to corresponding csv file.
            # should be done more often since fps rate decreases      
            if frame%100 == 0:
                write_pose_to_csv(id)
  
            end_time = time.time()
            print(f"FPS of video {id}: {1.0 / (end_time-start_time)}")
            frame += 1

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
    """
    This function tests CANDOR dataset. It searches the video files of every participiant (not the combined video, due to computation speed) and
    uses multithreading to analyze each of the videos. In the end, a csv file for every participiant exists, with the found pose coordinates
    in every frame.
    """
    all_files = get_all_movie_files()
    videolist = []
    for files_in_directory in all_files:
        videolist.append(files_in_directory[0]+"0")
        videolist.append(files_in_directory[1]+"1")
    
    print(videolist)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
       executor.map(analyze_video,videolist)

    
if __name__ == "__main__":

    # Assuring if GPU is used
    #print (torch.version.cuda)
    #print(torch.cuda.is_available())
    #print(torch.cuda.device_count())
    #print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))

    #test_emoreact()


    # landmark files available on https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
    base_options = python.BaseOptions(model_asset_path='./landmark_files/pose_landmarker_lite.task', delegate="GPU")
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,#LIVE_STREAM,
        num_poses=1,
        min_pose_detection_confidence=0.7,
        min_pose_presence_confidence=0.7,
        min_tracking_confidence=0.8,
        output_segmentation_masks=False,
        #result_callback=print_result   # only needed when RunningMode is LIVE_STREAM
    )
    test_candor()
