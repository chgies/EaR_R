import os
from dotenv import load_dotenv

CANDOR_DIR = None
load_dotenv()

# Get CANDOR PATH variable and standardize it
CANDOR_DIR = (os.getenv("CANDOR_DIR").replace("\\","/") + "/").replace("//","/")
print(f"Candor directory: {CANDOR_DIR}")

def get_candor_directory():
    """
    Return local path of CANDOR dataset
    
    Params:
        None
    
    Returns:
        CANDOR_DIR (String): The path of CANDOR dataset as String
    """
    return CANDOR_DIR

# def list all directories in a path
def list_directories(path):
    return [os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]
    

# get file with biggest file size in directory
def get_biggest_file(path):
    path += "/processed"
    movie_files = []
    for file in os.listdir(path):
        if file.endswith(".mp4"):
            movie_files.append(path + "/" + file)
    return max(movie_files, key=os.path.getsize)
    
# get the videos in directory and return them when they were not already analyzed
# (=> when there's no "extraction_completed" file for the movie)
def get_movie_files(path):
    path += "/processed/"
    movie_files = []
    for file in os.listdir(path):

        filename, ending = os.path.splitext(file)
        print(filename)
        if ending == ".mp4"and '-' not in filename:
            if not os.path.isfile(path + f"pose_extraction_of_{filename}_completed"):
                movie_files.append(path+file)
            else:
                print("1 file already completed")
    return movie_files

# for each dir get biggest file
def get_biggest_files():
    biggest_files = []
    for dir in dirs:
        completed = False
        for file in os.listdir(dir + "/processed/"):
            #print(file)
            if file.endswith("completed"):
                print("1 file already completed")
                completed = True

        if not completed:
            biggest_files.append(get_biggest_file(dir))
            
    return biggest_files

dirs = list_directories(CANDOR_DIR+'processed_dataset/processed/')
