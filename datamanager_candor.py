import os
import pandas as pd
from dotenv import load_dotenv

CANDOR_DIR = None
load_dotenv()

# Get CANDOR PATH variable and standardize it
CANDOR_DIR = (os.getenv("CANDOR_DIR").replace("\\","/") + "/").replace("//","/")
print(f"Candor directory: {CANDOR_DIR}")

# def list all directories in a path
def list_directories(path):
    return [os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]
    

# get file with biggest file size in directory
def get_biggest_file(path):
    path += "/processed"

    return max([os.path.join(path, o) for o in os.listdir(path)], key=os.path.getsize)

# get the videos in directory and return them when they were not already analyzed
# (=> when there's no "extraction_completed" file for the movie)
def get_movie_files(path):
    path += "/processed/"
    movie_files = []
    for file in os.listdir(path):
        filename, ending = os.path.splitext(file)
        if ending == ".mp4"and '-' not in filename:
            if not os.path.isfile(path + f"pose_extraction_of_{filename}_completed"):
                movie_files.append(path+file)
    return movie_files

# for each dir get biggest file
def get_biggest_files():
    return [get_biggest_file(dir) for dir in dirs]

# for each dir get movie files
def get_all_movie_files():
    return [get_movie_files(dir) for dir in dirs]

dirs = list_directories(CANDOR_DIR+'processed_dataset/processed/')
