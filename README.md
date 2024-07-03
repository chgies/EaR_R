This Repository is used for my BA Thesis, which is about emotion recognition in remote therapy situations.

Start pose extraction by downloading CANDOR dataset (see information on https://paperswithcode.com/dataset/candor-corpus-1), adding it to your PATH and execute run_pose_extraction.py.

Prerequisites:
Python >= 3.10
Python modules as mentioned in requirements.txt

Steps to start the extraction:

1. Add CANDOR directory to your PATH as CANDOR_DIR

2. Clone repository to your local file system

3. Install required python modules: "python -m pip install -r "./requirements.txt"

4. Run "run_pose_extraction.py" with parameters to start pose extraction for every video, all pose keypoint coordinates get extracted and saved as csv files into the directory you provide per command line. The file has following command line options:
    MANDATORY: '-p YOUR_POSE_DIRECTORY' or '--pose_dir YOUR_POSE_DIRECTORY': The directory where to put the csv files that contain the extracted poses (or where they are already saved, when you just want to calculate the features).
    '-fdir YOUR_FEATURE_DIRECTORY' or '--feature_dir YOUR_FEATURE_DIRECTORY': The directory where to put the csv file that contain the features calculated from the pose csv files.
    '-p' or '--extract_poses': Whether to extract poses out of the movies or not (if disabled and -fdir parameter is given, the features can be calculated without extracting the videos again).
    '-m MODEL_NAME' or '--model MODEL_NAME': The Pose Landmarker Model to use with MediaPipe. Possible: light, full and heavy. Default is heavy.
    '-w AMOUNT' or '--workers AMOUNT': help='The amount of parallel processes that are used to extract the dataset. Default is 1.
    '-fps' or '--show_fps': Show extraction speed while extracting the poses.Default is Off.
    '-v','--show_video': Show videos while extracting the poses. Disabling may slightly increase the pose extraction speed. Default is Off.

You can find information for feature extraction and further use in the 'references' folder

This code is work in progress.

Troubleshooting:
1) If MediaPipe Landmark Model initialization throws "RuntimeError: Can't open zip archive", download the 
   model from https://developers.google.com/mediapipe/solutions/vision/pose_landmarker and put it into the
   local "landmark_files" directory

