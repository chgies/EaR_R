import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import torch
import tensorflow as tf
from threading import Thread
import concurrent.futures
from urllib.request import urlretrieve
from pose_analyzer import PoseAnalyzer
from itertools import repeat

from datamanager_candor import get_all_movie_files
#from datamanager_candor import get_biggest_files
#from datamanager_emoreact import create_dataframe, get_df_test, get_df_train, get_df_val

landmark_type = 'lite'

def test_emoreact():
    df_test = get_df_test()
    df_train = get_df_train()
    df_val = get_df_val()
    print(df_test)
    print(df_train)
    print(df_val)
    for index, row in df_test.iterrows():
#        analyze_video(row['video'])
        print("done")

def test_candor(landmark_type):
    """
    This function tests CANDOR dataset. It searches the video files of every participiant (not the combined video, due to computation speed) and
    uses multithreading to analyze each of the videos. In the end, a csv file for every participiant exists, with the found pose coordinates
    in every frame.
    """

    # get a list of all videos that still have to be analyzed
    print("Gathering information about the video files in CANDOR dataset. Please wait...")
    all_files = get_all_movie_files()
    
    #all_files = ["/mnt/g/Abschlussarbeit_Datasets/CANDOR/processed_dataset/processed/fffda3e6-7d99-4db8-aa12-16e99fa454c2/processed/5d5162f1b50a1000169da137.mp4",
    #             "/mnt/g/Abschlussarbeit_Datasets/CANDOR/processed_dataset/processed/fffda3e6-7d99-4db8-aa12-16e99fa454c2/processed/5977e3867412f8000194e1fe.mp4"]
    videolist = []
    for files_in_directory in all_files:
        files_in_directory[0] = files_in_directory[0] + "_0"
        videolist.append(files_in_directory[0])
        
        files_in_directory[1] = files_in_directory[1] + "_1"
        videolist.append(files_in_directory[1])

    # start analyzing videos on video_list
    videos_to_analyze = len(videolist)
    print(f"{videos_to_analyze} videos still have to be analyzed. Working...")
    analyzed_videos = 0
    type = landmark_type
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        for result in executor.map(PoseAnalyzer, videolist, repeat(type)):
            if result != -1:
                analyzed_videos += 1
                print(f"{analyzed_videos} of {videos_to_analyze} videos have been analyzed.")

    
if __name__ == "__main__":

    # Assuring if GPU is used
    #print(torch.version.cuda)
    #print(torch.cuda.is_available())
    #print(torch.cuda.device_count())
    #print(f"CUDA-TF: {tf.test.is_built_with_cuda()}")
    #print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    #test_emoreact()

    # test if Mediapipe landmark task file is available
    landmark_type = "lite"
 
    test_candor(landmark_type)
