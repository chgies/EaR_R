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

    b. You can choose how many parallel processes should be started when extracting the coordinates by changing MAX_WORKERS varaible at line 24




####not yet implemented###

5. Run "candor_processing/models/CANDOR_model_training.py" to train your model
    a. You can choose which model you'd like to use by changing line 14 in CANDOR_model_training.py. If you type "EmotionV1", you train the model with features manually chosen by me. If you change it to "EmotionV50" or "EmotionV80" AND set line 21 "AUTO_SORT_IMPORTANCE" to True, the program will use Random Forest Classification to find the features with 50% resp. 80% importance and use them to train the net.
    b. You can chose to rescale the dataset by changin line 25 "CREATE_NORMALIZED_CSV" to True. The program will then use 3 normalization methods on the dataset, save the new datasets into files in the CANDOR train and test directories and train the normal and the normalized datasets sequentially. This can improve model accuracy.

6. Run "candor_test_model" to test the model with your webcam
    a. like in step 5, you can change line 15 "MODEL_TO_TEST" to choose the model you like to test. "EmotionV1", "EmotionV50" and "EmotionV80" are possible

7. If you want to use higher level Laban Motor Elements as features for model training, you can choose the USE_LABAN_FEATURES option:
    a. Set line 15 "USE_LABAN_FEATURES" to True in "run_pose_extraction.py" and run it. This will create csv files with Laban Motor Elements as features in CAER train, test and validation directories
    b. Then, set line 17 "USE_LABAN_FEATURES"  to True in "caer_processing/CAER_model_training.py" and run it to train a model with these csv files (option 5a is still possible)
    c. To test these new models, set line 19 "USE_LABAN_FEATURES" to True in "candor_test_model.py" and run it

### not yet implemented

You can find information for feature extraction and further use in the 'references' folder

This code is work in progress.

Troubleshooting:
1) If MediaPipe Landmark Model initialization throws "RuntimeError: Can't open zip archive", download the 
   model from https://developers.google.com/mediapipe/solutions/vision/pose_landmarker and put it into the
   local "landmark_files" directory