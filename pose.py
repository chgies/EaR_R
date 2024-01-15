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
from urllib.request import urlretrieve

#from datamanager_candor import get_all_movie_files
#from datamanager_candor import get_biggest_files
#from datamanager_emoreact import create_dataframe, get_df_test, get_df_train, get_df_val

output_window = None    # Used as screen variable to draw landmarks on
last_timestamp_ms = 0   # needed by Mediapipe when in "LIVE_STREAM"-RunningMode
frame = 0               # The current frame of the video, used to write pose data to csv

analyzed_results_person_0 = "frame,person,x,y,z\n"  # used to store pose data of the person with the id 0 in a conversation
analyzed_results_person_1 = "frame,person,x,y,z\n"  # used to store pose data of the person with the id 0 in a conversation


def check_pose_landmark_file(landmark_type):
    """
    This function checks if the needed Mediapipe Pose estimation landmark files are available in the sub directory
    '/landmark_files'. If not, they get downloaded from Mediapipe servers. 
    More info on https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

    Parameters:
    landmark_type (String): which of the current 3 models do you need for pose estimation. Can be 'lite', 'full' or 'heavy'

    Returns:
    The path to the file in string format
    """

    task_file_dir = './landmark_files/'
    match landmark_type:
        case 'lite':
            if os.path.isfile(task_file_dir + 'pose_landmarker_lite.task'):
                print("Mediapipe Pose Landmark file 'lite' already downloaded")
                return task_file_dir + 'pose_landmarker_lite.task'
            else:
                print("Mediapipe Pose Landmark file 'lite' not yet downloaded. Downloading now into './landmark_files' sub directory.")
                try:
                    url = (
                        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
                    )
                    filename = task_file_dir + "pose_landmarker_lite.task"
                    urlretrieve(url, filename)
                    print("Downloading Mediapipe Pose Landmark file 'pose_landmarker_lite.task' successful")
                    return task_file_dir + 'pose_landmarker_lite.task'
                except Exception as e:
                    print("An error occurred while downloading pose_landmarker_lite.task:")
                    print(e)
                    return -1
        case 'full':
            if os.path.isfile(task_file_dir + 'pose_landmarker_full.task'):
                print("Mediapipe Pose Landmark file 'full' already downloaded")
                return task_file_dir + 'pose_landmarker_full.task'
            else:
                print("Mediapipe Pose Landmark file 'full' not yet downloaded. Downloading now into './landmark_files' sub directory.")
                try:
                    url = (
                        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
                    )
                    filename = task_file_dir + "pose_landmarker_full.task"
                    urlretrieve(url, filename)
                    print("Downloading Mediapipe Pose Landmark file 'pose_landmarker_full.task' successful")
                    return task_file_dir + 'pose_landmarker_full.task'
                except Exception as e:
                    print("An error occurred while downloading pose_landmarker_full.task:")
                    print(e)
                    return -1
        case 'heavy':
            if os.path.isfile(task_file_dir + 'pose_landmarker_heavy.task'):
                print("Mediapipe Pose Landmark file 'heavy' already downloaded")
                return task_file_dir + 'pose_landmarker_heavy.task'
            else:
                print("Mediapipe Pose Landmark file 'heavy' not yet downloaded. Downloading now into './landmark_files' sub directory.")
                try:
                    url = (
                        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
                    )
                    filename = task_file_dir + "pose_landmarker_heavy.task"
                    urlretrieve(url, filename)
                    print("Downloading Mediapipe Pose Landmark file 'pose_landmarker_heavy.task' successful")
                    return task_file_dir + 'pose_landmarker_heavy.task'
                except Exception as e:
                    print("An error occurred while downloading pose_landmarker_heavy.task:")
                    print(e)
                    return -1
        case _:
            print("The given 'landmark_type' parameter is invalid. Can be 'lite', 'full' and 'heavy'") 
            print("More information on https://developers.google.com/mediapipe/solutions/vision/pose_landmarker")
            return -1

def write_pose_to_csv(file_path, person_id):
    """
    Writes the pose data of the person with the id person_id to the
    corresponding csv file. If no file exists, a new one is created. After
    storing the data, the global pose variable gets overwritten with "" to
    save memory and to speed up computation

    Parameters:
    file_path (String): the path of the video file whose pose data gets written
    person_id (Integer): The id of the person in the conversation, whose data gets written. Can be 0 or 1

    Returns:
    Nothing
    """
    global analyzed_results_person_0
    global analyzed_results_person_1
    
    # get the csv file name
    splitted_path = file_path.split("/")
    chars_to_cut_off = len(splitted_path[len(splitted_path)-1])
    file_path = file_path[:-chars_to_cut_off]
    file_name = file_path + splitted_path[len(splitted_path)-3]+ f"_posedata_{person_id}.csv"
    
    # actually write to the csv file
    if os.path.isfile(file_name):
        file = open(file_name, 'a', newline='')
        if person_id == 0:
            if analyzed_results_person_0.startswith("frame,person,x,y,z"):
                analyzed_results_person_0 = analyzed_results_person_0.lstrip("frame,person,x,y,z\n")
            file.writelines(analyzed_results_person_0)
            analyzed_results_person_0 = ""
        if person_id == 1:
            if analyzed_results_person_1.startswith("frame,person,x,y,z"):
                analyzed_results_person_1 = analyzed_results_person_1.lstrip("frame,person,x,y,z\n")
            file.writelines(analyzed_results_person_1)
            analyzed_results_person_1 = ""
    else:
        file = open(file_name, 'w', newline='')
        if person_id == 0:
            file.writelines(analyzed_results_person_0)
            analyzed_results_person_0 = ""
        if person_id == 1:
            file.writelines(analyzed_results_person_1)
            analyzed_results_person_1 = ""
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

def draw_landmarks(image, person_id, results):
    """
    Draws landmarks on the given image using the results from pose estimation.

    Args:
        image (numpy.ndarray): The input image.
        person_id (int): The ID of the person of the conversation whose video is processed. Can be 0 or 1 
        results (mediapipe.python.solution_base.SolutionOutputs): The pose estimation results.

    Returns:
        numpy.ndarray: The image with landmarks drawn.
    """
    
    pose_landmarks_list = results.pose_landmarks

    annotated_image = np.copy(image)
    global frame

    global analyzed_results_person_0
    global analyzed_results_person_1

    # draw poses on opencv window.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
            
        # this is used to save found points for csv files
        for landmark in pose_landmarks:
            new_row = f"{frame},{person_id},{landmark.x},{landmark.y},{landmark.z}"
            if person_id == 0:
                analyzed_results_person_0 += "\n" + new_row
            else:
                analyzed_results_person_1 += "\n" + new_row    
                
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


def analyze_video(path):
    """
    Analyzes a video file or webcam stream and detects human poses using MediaPipe Pose. 
    First step is to get the id of that video in it's directory, i.e. if it is the video of
    the first person or the second ( id = 0 or 1).

    Args:
        path (str): The path to the video file or '0' for webcam stream.
    Returns:
        None
    """

    # get id of the video, then delete the id ending for further processing
    id_ending_split = path.split("_")
    if id_ending_split[len(id_ending_split)-1] == "0":
        id = 0
    else:
        id = 1
    path = path[:-2]
    print(f"Starting analysis of video {path}")

    with vision.PoseLandmarker.create_from_options(options) as landmarker:        
        cap = cv2.VideoCapture(path)
        global frame
        frame = 0
        while cap.isOpened():
            success, image = cap.read()
            start_time = time.time()
            if not success:
                
                # create a "completed" file for pausing analysis of CANDOR dataset when movie file is over.    
                if frame > 0:
                    splitted_path = path.split("/")
                    chars_to_cut_off = len(splitted_path[len(splitted_path)-1])
                    completed_path = path[:-chars_to_cut_off]
                    file_name = completed_path + "pose_extraction_of_" + "{splitted_path[len(splitted_path)-1]}".rstrip(".mp4") + "_completed"
                    file = open(file_name, 'w', newline='')
                    file.close()
                break
            
            if not detect_both_online(image):
                frame += 1
                continue

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
                      
            output_window = cv2.cvtColor(
                draw_landmarks(image, id, result), cv2.COLOR_RGB2BGR)
            
            if output_window is not None:
                cv2.imshow("MediaPipe Pose Landmark", output_window)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return -1

            # write found pose data every x frames to corresponding csv file.
            # should be done more often since fps rate decreases      
            if frame%100 == 0:
                write_pose_to_csv(path, id)
  
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

    # get a list of all videos that still have to be analyzed
    #all_files = get_all_movie_files()
    all_files = ["/mnt/g/Abschlussarbeit_Datasets/CANDOR/processed_dataset/processed/fffda3e6-7d99-4db8-aa12-16e99fa454c2/processed/5d5162f1b50a1000169da137.mp4",
                 "/mnt/g/Abschlussarbeit_Datasets/CANDOR/processed_dataset/processed/fffda3e6-7d99-4db8-aa12-16e99fa454c2/processed/5977e3867412f8000194e1fe.mp4"]
    videolist = []
    for files_in_directory in all_files:
        if all_files.index(files_in_directory)%2 == 0:
            files_in_directory = files_in_directory + "_0"
            videolist.append(files_in_directory)
        else:
            files_in_directory = files_in_directory + "_1"
            videolist.append(files_in_directory)
    
    # start analyzing videos on video_list
    videos_to_analyze = len(videolist)
    print(f"{videos_to_analyze} videos still have to ba analyzed. Working...")
    analyzed_videos = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        for result in executor.map(analyze_video,videolist):
            if result != -1:
                analyzed_videos += 1
                print(f"{analyzed_videos} out of {videos_to_analyze} videos hav been analyzed.")

    
if __name__ == "__main__":

    # Assuring if GPU is used
    #print (torch.version.cuda)
    #print(torch.cuda.is_available())
    #print(torch.cuda.device_count())
    #print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))

    #test_emoreact()

    # test if Mediapipe landmark task file is available
    landmark_type = 'lite'
    landmark_path = check_pose_landmark_file(landmark_type)
    print(landmark_path)
    # define Mediapipe pose landmark detector's options
    base_options = python.BaseOptions(model_asset_path=landmark_path, delegate="GPU")
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.7,
        min_pose_presence_confidence=0.7,
        min_tracking_confidence=0.8,
        output_segmentation_masks=False,
    )
    #test_candor()
