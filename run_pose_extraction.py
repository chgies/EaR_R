import os
import argparse
import concurrent.futures
import torch
import tensorflow as tf
from candor_processing.candor_pose_analyzer import CANDORPoseAnalyzer
from candor_processing.datamanager_candor import get_candor_movie_files
from candor_processing.run_candor_feature_extraction import extract_all_csv_values
# Choose if you want to train the net with features following 
# Aristidou (2015, aee references folder), or high level Laban motor elements
# not yet implemented!!!
USE_LABAN_FEATURES = False

CANDOR_DIR = os.environ["CANDOR_DIR"]
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def extract_candor(landmark_type, dir_to_extract, max_workers):
    """
    This function tests CANDOR dataset. It searches the video files of every participiant (not the combined video, due to computation speed) and
    uses multiprocessing to analyze each of the videos. In the end, a csv file for every video exists, with the found pose coordinates
    in every frame. 
    
    Params:
        landmark_type (String): The MediaPipe Pose Landmarker model which is used for pose estimation
        dir_to_extract (String): The directory where the csv files containing the pose coordinates are saved
        max_workers (int): The amount of parallel processes started to extract the videos
    Returns:
        None
    """
    
    # get a list of all videos that still have to be analyzed
    print("Gathering information about the video files in CANDOR dataset. Please wait...")
    videolist = get_candor_movie_files(CANDOR_DIR)
    
    # for testing purposes use this line instead of the former line (on windows, on linux change path accordingly):
    #videolist = [f"{CANDOR_DIR}/processed_dataset/processed/fffda3e6-7d99-4db8-aa12-16e99fa454c2/processed/5d5162f1b50a1000169da137.mp4", f"{CANDOR_DIR}/processed_dataset/processed/fffda3e6-7d99-4db8-aa12-16e99fa454c2/processed/5977e3867412f8000194e1fe.mp4"]
    # start analyzing videos on video_list
    videos_to_analyze = len(videolist)
    print(f"{videos_to_analyze} videos still have to be analyzed. Working...")
    analyzed_videos = 0
    candorPoseAnalyzer = CANDORPoseAnalyzer(landmark_type, dir_to_extract)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(candorPoseAnalyzer.analyze_video, videolist):
            if result != -1:
                analyzed_videos += 1
                print(f"{analyzed_videos} of {videos_to_analyze} videos have been analyzed.")
    print("All pose coordinates have been extracted from CANDOR dataset.")
    print("Extracting the Laban features and components from found pose data. Please Wait...")

    extract_all_csv_values(USE_LABAN_FEATURES, dir_to_extract)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This program uses MediaPipeto to extract pose coordinates out of the CANDOR dataset.')
    parser.add_argument('-d','--extraction_dir', help='The directory where to put the csv files that contain the extracted poses', required=True)
    parser.add_argument('-m','--model', help='The Pose Landmarker Model to use with MediaPipe. Possible: lite, full and heavy. Default is heavy', required=False)
    parser.add_argument('-w','--workers', help='The amount of parallel processes that are used to extract the dataset. Default is 1', required=False)
    parser.parse_args()
    args = vars(parser.parse_args())    
    dir_to_extract = args['extraction_dir']
    if args['model'] == 'lite':
        landmark_type = 'lite'
    elif args['model'] == 'full':
        landmark_type = 'full'
    if args['model'] == 'heavy':
        landmark_type = 'heavy'
    else:
        landmark_typeEL_TO_CHOOSE = 'heavy'
    if not args['workers'] == None:
        max_workers = int(args['workers'])
    else:
        max_workers = 1

    # Assuring if GPU is used
    print("Checking CUDA and GPU support:")
    print(f"    PyTorch version built with CUDA support: {torch.version.cuda}")
    print(f"    CUDA is available for PyTorch: {torch.cuda.is_available()}")
    print(f"    Found CUDA devices: {torch.cuda.device_count()}")
    print(f"    Tensorflow found devices: {tf.config.list_physical_devices('GPU')}")

    # start the actual extraction    
    extract_candor(landmark_type, dir_to_extract, max_workers)
