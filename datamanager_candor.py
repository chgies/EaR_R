import os
import pandas as pd
from dotenv import load_dotenv

CANDOR_DIR = None
load_dotenv()
CANDOR_DIR = os.getenv("CANDOR_DIR")
print(CANDOR_DIR)

# def list all directories in a path
def list_directories(path):
    return [os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]

# get file with biggest file size in directory
def get_biggest_file(path):
    path += "/processed"

    # changed this path to fit for my directory structure
    return max([os.path.join(path, o) for o in os.listdir(path)], key=os.path.getsize)+'/processed/fffda3e6-7d99-4db8-aa12-16e99fa454c2.mp4'

# for each dir get biggest file
def get_biggest_files():
    return [get_biggest_file(dir) for dir in dirs]


dirs = list_directories(CANDOR_DIR)
