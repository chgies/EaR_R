This Repository is used for my BA Thesis, which is about Emotion recognition in remote therapy situations.

Start pose extraction by downloading CANDOR dataset (see information on https://paperswithcode.com/dataset/candor-corpus-1), adding it to your PATH and execute run_pose_extraction.py.

Prerequisites:
Python >= 3.10
Python modules as mentioned in requirements.txt

Steps to start the extraction:

1. Add CANDOR directory to your PATH as CANDOR_DIR

2. Clone repository to your local file system

3. If needed, install required python modules: "python -m pip install -r "./requirements.txt"

4. Run "run_pose_extraction.py" to start pose extraction for every video, all pose keypoint coordinates get extracted and saved as csv files in CANDOR directory, creating a new "extracted_coordinates" directory
    a. You can choose which Mediapipe posel model you like to use by changing line 17 in "run_pose_extraction.py". Possible models are "lite", "full", "heavy". You can find more information at https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

    b. You can choose how many parallel processes should be started when extracting the coordinates by changing MAX_WORKERS variable at line 24

You can find information for feature extraction and further use in the 'references' folder

This code is work in progress.

Troubleshooting:
1) If MediaPipe Landmark Model initialization throws "RuntimeError: Can't open zip archive", download the 
   model from https://developers.google.com/mediapipe/solutions/vision/pose_landmarker and put it into the
   local "landmark_files" directory

