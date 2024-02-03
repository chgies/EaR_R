import os
import pandas as pd
from dotenv import load_dotenv

CAER_DIR = None
load_dotenv()

# Get CAER PATH variable and standardize it (for win/linux)
CAER_DIR = (os.getenv("CAER_DIR").replace("\\","/") + "/").replace("//","/")
print(f"CAER directory: {CAER_DIR}")

def get_caer_movie_files(CAER_DIR):
    """
    This function recursively searches for .avi files in a directory and
    returns a list
    
    Parameters:
        CAER_DIR (String): The directory path of CAER dataset
    
    Returns:
        movie_list (List): A list of paths to all found .avi files
    """
    movie_list = []
    for r, d, f in os.walk(CAER_DIR):
        for file in f:
            if '.avi' in file:
                movie_list.append(os.path.join(r, file))
    return movie_list