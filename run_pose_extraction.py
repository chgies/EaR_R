import torch
import concurrent.futures
from caer_processing.caer_pose_analyzer import CAERPoseAnalyzer
from caer_processing.datamanager_caer import get_caer_movie_files
from caer_processing.datamanager_caer import get_caer_directory
from caer_processing.run_caer_feature_extraction import extract_all_csv_values
from itertools import repeat
import os

# here you can define the landmark pose extraction model of mediapipe
# Possible values: 'lite', 'full', 'heavy'
MEDIAPIPE_MODEL_TO_CHOOSE = 'heavy'

# Choose if you want to train the net with features following Aristidou (2015, aee references folder), or high level Laban motor elements
USE_LABAN_FEATURES = False

CAER_DIR = os.environ["CAER_DIR"]
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def test_caer(landmark_type):
    """
    This function tests CAER dataset. It searches the video files of every participiant (not the combined video, due to computation speed) and
    uses multiprocessing to analyze each of the videos. In the end, a csv file for every video exists, with the found pose coordinates
    in every frame. You can change the max amount of concurrent processes by changing the 'Max_workers' value in the
    ProcessPoopExecutor - calling line
    """
    
    # get a list of all videos that still have to be analyzed
    print("Gathering information about the video files in CAER dataset. Please wait...")
    videolist = get_caer_movie_files(get_caer_directory())
    
    # for testing purposes use this line instead of the former line (on windows, on linux change path accordingly):
    videolist = [f"{CAER_DIR}/test/Anger/0001.avi"]
    # start analyzing videos on video_list
    videos_to_analyze = len(videolist)
    print(f"{videos_to_analyze} videos still have to be analyzed. Working...")
    analyzed_videos = 0
    type = landmark_type
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        for result in executor.map(CAERPoseAnalyzer, videolist, repeat(type)):
            if result != -1:
                analyzed_videos += 1
                print(f"{analyzed_videos} of {videos_to_analyze} videos have been analyzed.")
    print("All pose coordinates have been extracted from CAER dataset.")
    print("Extracting the Laban features and components from found pose data. Please Wait...")
    
    #extract_all_csv_values(USE_LABAN_FEATURES)
    
if __name__ == "__main__":

    # Assuring if GPU is used
    print(f"PyTorch version built with CUDA support: {torch.version.cuda}")
    print(f"CUDA is available for PyTorch: {torch.cuda.is_available()}")
    print(f"Found CUDA devices: {torch.cuda.device_count()}")
    
    landmark_type = MEDIAPIPE_MODEL_TO_CHOOSE
 
    test_caer(landmark_type)
