This Repository is used for my BA Thesis, which is about Emotion and Rapport recognition in remote therapy situations.

Start pose extraction by downloading the CANDOR and CAER datasets, adding them to your PATH and execute pose_processing.py

Prerequisites:
Python >= 3.10
Python modules as mentioned in requirements.txt

Steps to start the extraction:
1. Add CANDOR directory to your PATH as CANDOR_DIR
2. Add CAER directory to your PATH as CAER_DIR
3. Clone repository to your local file system 
4. run "pose_processing.py" to start pose extraction for every videos of these datasets, they get saved in csv files in ever video directory
5. (NOT FULLY IMPLEMENTED, SEE STEP 5a): start "caer_processing/caer_feature_extractor.py" to extract Laban elements out of every csv file
    5a. Element extraction is not yet implemented for every video file. Currently the "caer_feature_extractor.py" only uses the local "CAER_pose_example.csv" file

Currently implemented:
    - pose extraction from CANDOR into csv files
    - pose extraction of CAER into csv files
    - skeleton functions for doing extractions in a cloud environment
    - feature extraction for CAER csv files
    - Laban element calculation for CAER files

ToDo:
    - create system to extract Laban elements out of every video csv file
    - bundle the dataframes for every video of an emotion into 1 csv file
    - Use these dataframes to get an overview of the specific values for every emotion
    - Use these dataframes to calculate Laban components for the elements
    - Train a Neural Network ising the pose data, the elements (and the emotions as labels)
    - To all of this again for CANDOR dataset
    - WRITE TESTS, CLEAN UP!

You can find information for feature extraction and further use in the 'references' folder

This code is work in progress.
