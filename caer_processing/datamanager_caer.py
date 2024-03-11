import os

CAER_DIR = None

# Get CAER PATH variable and standardize it (make it usable regardless of windows or linux os)
CAER_DIR = os.environ["CAER_DIR"]
print(f"CAER directory: {CAER_DIR}")

def get_caer_directory():
    """
    Return local path of CAER dataset
    
    Params:
        None
    
    Returns:
        CAER_DIR (String): The path of CAER dataset as String
    """
    return CAER_DIR

def get_caer_movie_files(CAER_DIR):
    """
    This function recursively searches for .avi files in CAER directory and
    returns a list
    
    Params:
        CAER_DIR (String): The directory path of CAER dataset
    
    Returns:
        movie_list (List): A list of paths to all found .avi files
    """
    movie_list = []
    for r, d, f in os.walk(CAER_DIR):
        for file in f:
            if str(file).endswith('.avi'):
                movie_list.append(os.path.join(r, file))
    return movie_list

def get_caer_csv_files(CAER_DIR):
    """
    This function recursively searches for .csv files in CAER directory and
    returns a list
    
    Parameters:
        CAER_DIR (String): The directory path of CAER dataset
    
    Returns:
        csv_list (List): A list of paths to all found .csv files
    """
    csv_list = []
    for r, d, f in os.walk(CAER_DIR):
        for file in f:
            if '.csv' in file:
                csv_list.append(os.path.join(r, file))
    return csv_list