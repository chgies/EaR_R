import os
import json

# Filepath of extracted dataset
DATASET_FILEPATH = 'E:/Abschlussarbeit_Datasets/CANDOR/processed_dataset/data/00ae2f18-9599-4df6-8e3a-6936c86b97f0'
# user ids of dataset, used for mp4 filename
video_user_ids = json.load(open(DATASET_FILEPATH + '/processed/channel_map.json'))
print(video_user_ids)
video_paths = [DATASET_FILEPATH + '/processed/' + video_user_ids.get('L') + '.mp4']#, filepath + '/processed/' + user_names.get('R') + '.mp4']
print(video_paths)
output_csv_paths = [DATASET_FILEPATH + '/processed/mp_data/' + video_user_ids.get('L') + '.csv']#, filepath + '/processed/mp_data/' + user_names.get('R') + '.csv']

def get_video_paths():
    return video_paths

def get_csv_output_paths():
    return output_csv_paths