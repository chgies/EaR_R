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

def extract_candor(extract_poses, landmark_type, pose_dir_to_extract, feature_dir_to_extract, max_workers, show_fps, show_video):
    """
    This function tests CANDOR dataset. It searches the video files of every participiant (not the combined video, due to computation speed) and
    uses multiprocessing to analyze each of the videos. In the end, a csv file for every video exists, with the found pose coordinates
    in every frame. 
    
    Params:
        extract_poses (Boolean): A variable to define if the poses have to be extracted out of the video
        landmark_type (String): The MediaPipe Pose Landmarker model which is used for pose estimation
        pose_dir_to_extract (String): The directory where the csv files containing the pose coordinates are saved
        feature_dir_to_extract (String): The directory where the csv file containing the features calculated from the pose coordinates is saved
        max_workers (int): The amount of parallel processes started to extract the videos
        show_fps (Boolean): A variable to define if the current extraction speed is printed to the command line (in Frames per second)
        show_video (Boolean): A variable to define if the candor videos are opened and poses are drawn to them.
    Returns:
        None
    """

 
    if extract_poses:
        if not os.path.exists(pose_dir_to_extract):
            os.makedirs(pose_dir_to_extract)
        already_extracted_file = f"{pose_dir_to_extract}/extracted_videos.txt"
        if os.path.isfile(already_extracted_file):
            file = open(already_extracted_file, 'r')
            already_extracted_list = file.read().splitlines()
            file.close
        else:
            already_extracted_list = []
        # get a list of all videos that still have to be analyzed
        print("Gathering information about the video files in CANDOR dataset. Please wait...")
        videolist = get_candor_movie_files(CANDOR_DIR)
        # for testing purposes use this line instead of the former line (on windows, on linux change path accordingly):
        #videolist = [f"{CANDOR_DIR}/processed_dataset/processed/fffda3e6-7d99-4db8-aa12-16e99fa454c2/processed/5d5162f1b50a1000169da137.mp4", f"{CANDOR_DIR}/processed_dataset/processed/fffda3e6-7d99-4db8-aa12-16e99fa454c2/processed/5977e3867412f8000194e1fe.mp4"]
        clean_video_list = []
        for video in videolist:
            if os.path.split(video)[1] not in already_extracted_list:
                clean_video_list.append(video)
        
        # start analyzing videos on video_list
        videos_to_analyze = len(clean_video_list)
        print(f"{videos_to_analyze} videos still have to be analyzed. Working...")
        analyzed_videos = 0
        candorPoseAnalyzer = CANDORPoseAnalyzer(landmark_type, pose_dir_to_extract, show_fps, show_video)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(candorPoseAnalyzer.analyze_video, clean_video_list):
                if result != -1:
                    analyzed_videos += 1
                    print(f"{analyzed_videos} of {videos_to_analyze} videos have been analyzed.")
        print("All pose coordinates have been extracted from CANDOR dataset.")
        
    if not feature_dir_to_extract == "":
        print("Extracting the Laban features and components from found pose data. Please Wait...")
        extract_all_csv_values(USE_LABAN_FEATURES, pose_dir_to_extract, feature_dir_to_extract)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This program uses MediaPipeto to extract pose coordinates out of the CANDOR dataset.')
    parser.add_argument('-p','--extract_poses', help='Set if you want to extract poses out of the CANDOR Dataset. Default is False', action='store_true', required=False)
    parser.add_argument('-pdir','--pose_dir', help='MANDATORY. The directory where the csv files are located, that contain the extracted poses. If extract_poses parameter is set, pose files are extracted to this directory', required=True)
    parser.add_argument('-fdir','--feature_dir', help='The directory where to put the csv file that contains the extracted features. If not specified, features wont be extracted', required=False)
    parser.add_argument('-m','--model', help='The Pose Landmarker Model to use with MediaPipe. Possible: lite, full and heavy. Default is heavy', required=False)
    parser.add_argument('-w','--workers', help='The amount of parallel processes that are used to extract the dataset. Default is 1', required=False)
    parser.add_argument('-fps','--show_fps', help='Show extraction speed while extracting the poses. Default is Off',action='store_true', required=False)
    parser.add_argument('-v','--show_video', help='Show videos while extracting the poses. Disabling may slightly increase the pose extraction speed. Default is Off',action='store_true', required=False)
    parser.parse_args()
    args = vars(parser.parse_args()) 
    
    if args['extract_poses'] is False:
        extract_poses = False
        print("User chose to not extract Poses out of the CANDOR video files")
    else:
        extract_poses = True
        print("User chose to extract Poses out of the CANDOR video files")
    pose_dir_to_extract = args['pose_dir']
    if args['feature_dir'] is None:
        feature_dir_to_extract = ""
        print("User chose to not extract features out of the pose csv files")
    else:
        feature_dir_to_extract = args['feature_dir']
        print("User chose to extract features out of the pose csv files")
    if args['model'] == 'lite':
        landmark_type = 'lite'
    elif args['model'] == 'full':
        landmark_type = 'full'
    if args['model'] == 'heavy':
        landmark_type = 'heavy'
    else:
        landmark_type = 'heavy'
    if args['workers'] is None:
        max_workers = 1
    else:
        max_workers = int(args['workers'])
    if args['show_fps'] is False:
        show_fps = False
    else:
        show_fps = True
    if args['show_video'] is False:
        show_video = False
    else:
        show_video = True
    
    # Assuring if GPU is used
    print("Checking CUDA and GPU support:")
    print(f"    PyTorch version built with CUDA support: {torch.version.cuda}")
    print(f"    CUDA is available for PyTorch: {torch.cuda.is_available()}")
    print(f"    Found CUDA devices: {torch.cuda.device_count()}")
    print(f"    Tensorflow found devices: {tf.config.list_physical_devices('GPU')}")

    # start the actual extraction    
    extract_candor(extract_poses, landmark_type, pose_dir_to_extract, feature_dir_to_extract, max_workers, show_fps, show_video)
