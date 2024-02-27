This Repository is used for my BA Thesis, which is about Emotion and Rapport recognition in remote therapy situations.

Start pose extraction by downloading the CANDOR and CAER datasets, adding them to your PATH and execute pose_processing.py

Prerequisites:
Python >= 3.10
Python modules as mentioned in requirements.txt

Steps to start the extraction:
1. Add CANDOR directory to your PATH as CANDOR_DIR
2. Add CAER directory to your PATH as CAER_DIR
3. Clone repository to your local file system 
4. Run "pose_processing.py" to start pose extraction for every videos of these datasets, they get saved in csv files in ever video directory
5. Start "caer_processing/run_feature_extraction.py" to extract Laban elements out of every csv file
6. (STARTED) Run "caer_processing/models/CAER_model_training.py" to train your model
7. Run "caer_processing/caer_test_model" to test the model with your webcam

Currently implemented:
    - pose extraction from CANDOR into csv files
    - pose extraction of CAER into csv files
    - skeleton functions for doing extractions in a cloud environment
    - feature extraction for CAER csv files
    - Laban element calculation for CAER files
    - Use these dataframes to get an overview of the specific values for every emotion
    - Use these dataframes to calculate Laban components for the elements
    - Train a Neural Network ising the pose data, the elements (and the emotions as labels) (STARTED)
    
ToDo:
    - To all of this again for CANDOR dataset
    - WRITE TESTS, CLEAN UP!

You can find information for feature extraction and further use in the 'references' folder

This code is work in progress.
