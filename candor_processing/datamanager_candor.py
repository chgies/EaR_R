import os

def get_candor_movie_files(candor_dir):
    """
    This function recursively searches for small (=short-named) .mp4 files in CANDOR directory and
    returns a list. 
    
    Params:
        CANDOR_DIR (String): The directory path of CANDOR dataset
    
    Returns:
        movie_list (List): A list of paths to all found .avi files
    """
    movie_list = []
    for r, d, f in os.walk(candor_dir):
        for file in f:
            if str(file).endswith('.mp4') and "-" not in str(file):
                # Check if corresponding 'audio_video_features.csv' file exists and is not empty; for emotion extraction
                    csv_file = f"{r[:-10]}/audio_video_features.csv"
                    if os.path.exists(csv_file) and os.stat(csv_file).st_size != 0:
                        movie_list.append(os.path.join(r, file))
    return movie_list

def get_candor_feature_csv_files(CANDOR_DIR):
    """
    This function recursively searches for audio_video_features.csv files in CANDOR directory and
    returns a list
    
    Parameters:
        CANDOR_DIR (String): The directory path of CANDOR dataset
    
    Returns:
        csv_list (List): A list of paths to all found .csv files
    """
    csv_list = []
    for r, d, f in os.walk(CANDOR_DIR):
        for file in f:
            if '.csv' in file and file.endswith("audio_video_features.csv"):
                csv_list.append(os.path.join(r, file))
    return csv_list

def get_candor_pose_csv_files(pose_csv_dir):
    """
    This function recursively searches for the csv files in the directory given by the user and
    returns a list
    
    Parameters:
        pose_csv_dir (String): The directory path of where the extracted pose csv files are located
    
    Returns:
        csv_list (List): A list of paths to all found .csv files
    """
    csv_list = []
    for r, d, f in os.walk(pose_csv_dir):
        for file in f:
            if file.endswith("posedata.csv"):
                csv_list.append(os.path.join(r, file))
    return csv_list