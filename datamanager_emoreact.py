import os
import pandas as pd
from dotenv import load_dotenv

EMOREACT_DIR = None
load_dotenv()
EMOREACT_DIR = os.getenv("EMOREACT_DIR")
print(EMOREACT_DIR)

directory_test = EMOREACT_DIR+"/Data/Test"
directory_train = EMOREACT_DIR+"/Data/Train"
directory_val = EMOREACT_DIR+"/Data/Validation"

# create pandas dataframe from directory with video files and labels txt, first column is video file name other columns are from labels txt
# 1- Curiosity
# 2- Uncertainty
# 3- Excitement
# 4- Happiness
# 5- Surprise
# 6- Disgust
# 7- Fear
# 8- Frustration
# 9- Valence

def create_dataframe(directory, label_file):
    df = pd.DataFrame(columns=['video', 'Curiosity', 'Uncertainty', 'Excitement', 'Happiness', 'Surprise', 'Disgust', 'Fear', 'Frustration', 'Valence'])
    file = open(label_file)
    for filename in os.listdir(directory):
        full_path = os.path.join(directory, filename)
        if filename.endswith(".mp4"):
            line = file.readline()
            splits = line.split(",")
            df = df._append({'video': full_path,'Curiosity': splits[0], 'Uncertainty': splits[1], 'Excitement': splits[2], 'Happiness': splits[3], 'Surprise': splits[4], 'Disgust': splits[5], 'Fear': splits[6], 'Frustration': splits[7], 'Valence': splits[8]}, ignore_index=True)
    return df

def get_df_test():
    return create_dataframe(directory_test, EMOREACT_DIR+"/Labels/test_labels.text")

def get_df_train():
    return create_dataframe(directory_train, EMOREACT_DIR+"/Labels/train_labels.text")

def get_df_val():
    return create_dataframe(directory_val, EMOREACT_DIR+"/Labels/val_labels.text")
