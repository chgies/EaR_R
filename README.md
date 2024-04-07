This Repository is used for my BA Thesis, which is about Emotion recognition in remote therapy situations.

Start pose extraction by downloading CAER dataset, adding them to your PATH and execute pose_processing.py

Prerequisites:
Python >= 3.10
Python modules as mentioned in requirements.txt

Steps to start the extraction:
1. Add CAER directory to your PATH as CAER_DIR
2. Clone repository to your local file system
3. If needed, install required python modules: "python -m pip install -r "./requirements.txt"
4. Run "run_pose_extraction.py" to start pose extraction for every video, all poses get extracted, recalculated and saved as csv files in CAER "train", "test" and "validation" directories
    a. You can choose which Mediapipe posel model you like to use by changing line 12 in "run_pose_extraction.py". Possible models are "lite", "full", "heavy". You can find more information at https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

5. Run "caer_processing/models/CAER_model_training.py" to train your model
    a. You can choose which model you'd like to use by changing line 16 in CAER_model_training.py. If you type "EmotionV1", you train the model with features manually chosen by me. If you change it to "EmotionV50" or "EmotionV80" AND set line 13 "AUTO_SORT_IMPORTANCE" to True, the program will use Random Forest Classification to find the features with 50% resp. 80% importance anf use them to train the net.
    b. You can chose to rescale the dataset by changin line 14 "CREATE_NORMALIZED_CSV" to True. The program will then use 3 normalization methods on the dataset, save the new datasets into files in the CAER trian and test directories and train the normal and the normalized datasets sequentially. This can improve model accuracy.
6. Run "caer_test_model" to test the model with your webcam
    a. like in step 5, you can change line 15 "MODEL_TO_TEST" to choose the model you like to test. "EmotionV1", "EmotionV50" and "EmotionV80" are possible

Currently implemented:
    - pose extraction from CANDOR into csv files
    - pose extraction of CAER into csv files
    - skeleton functions for doing extractions in a cloud environment
    - feature extraction for CAER csv files
    - Laban element calculation for CAER files
    - Use these dataframes to get an overview of the specific values for every emotion
    - Use these dataframes to calculate Laban components for the elements
    - Train a Neural Network using the pose data, the elements (and the emotions as labels) (STARTED)
    
ToDo:
    - WRITE TESTS, CLEAN UP!

You can find information for feature extraction and further use in the 'references' folder

This code is work in progress.

Troubleshooting:
1) If MediaPipe Landmark Model initialization throws "RuntimeError: Can't open zip archive", download the 
   model from https://developers.google.com/mediapipe/solutions/vision/pose_landmarker and put it into the
   local "landmark_files" directory