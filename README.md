This Repository is used for my BA Thesis, which is about emotion recognition in remote therapy situations.

Start pose extraction by downloading CANDOR dataset (see information on https://paperswithcode.com/dataset/candor-corpus-1), adding it to your PATH and execute run_pose_extraction.py.

Prerequisites:
Python >= 3.10
Python modules as mentioned in requirements.txt

Steps to start the extraction:

1. Add CANDOR directory to your PATH as CANDOR_DIR

2. Clone repository to your local file system

3. Install required python modules: "python -m pip install -r "./requirements.txt"

4. Run "run_pose_extraction.py" to start pose extraction for every video, all pose keypoint coordinates get extracted and saved as csv files into the directory you provide per command line. The file has following command line options:
    MANDATORY: '-d' or '--extraction_dir': The directory where to put the csv files that contain the extracted poses
    '-m' or '--model': The Pose Landmarker Model to use with MediaPipe. Possible: light, full and heavy. Default is heavy
    -w' or '--workers': help='The amount of parallel processes that are used to extract the dataset. Default is 1

You can find information for feature extraction and further use in the 'references' folder

This code is work in progress.

Troubleshooting:
1) If MediaPipe Landmark Model initialization throws "RuntimeError: Can't open zip archive", download the 
   model from https://developers.google.com/mediapipe/solutions/vision/pose_landmarker and put it into the
   local "landmark_files" directory

