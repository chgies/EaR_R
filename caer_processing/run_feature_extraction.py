import os
import pandas as pd
import concurrent.futures
from datamanager_caer import get_caer_csv_files
from datamanager_caer import get_caer_directory
from caer_feature_extractor import CAERFeatureExtractor


def extract_values_in_dir(csv_directory):
    """
    Scan all csv files in a given directory of the CAER dataset, extract the calculated values and combine them all into one big csv file to
    make this data available for model training. After that, add a column with "emotion" label.

        Params: 
            csv_dorectory (String): The path to the directory in CAER dataset
        Returns:
            None
    """
    combined_dataframe = pd.DataFrame(columns=['f2_min', 'f2_max', 'f2_mean', 'f2_std', 'f3_min', 'f3_max', 'f3_mean', 'f3_std', 'f4_min', 'f4_max', 'f4_mean', 'f4_std', 'f5_min', 'f5_max', 'f5_mean', 'f5_std', 'f8_min', 'f8_max', 'f8_mean', 'f8_std', 'f10_min', 'f10_max', 'f10_mean', 'f11_num_peaks', 'f12_min', 'f12_max', 'f12_std', 'f13_min', 'f13_max', 'f13_std', 'f15_min', 'f15_std', 'f16_min', 'f16_std', 'f18_min', 'f18_std', 'f19_min', 'f19_max', 'f19_mean', 'f19_std', 'f20_min', 'f20_max', 'f20_mean', 'f20_std', 'f22_min', 'f22_max', 'f22_mean', 'f22_std', 'f23_min', 'f23_max', 'f23_mean', 'f23_std', 'f24_min', 'f24_max', 'f24_mean', 'f24_std', 'f25_mean', 'emotion'])
    extracted_file_index = 0
    csv_file_list = get_caer_csv_files(csv_directory)
    csv_dir_list = []
    for csv_file in csv_file_list:
        splitted_path = csv_file.split("/")
        if "/".join(splitted_path[:len(splitted_path)-1]) not in csv_dir_list:
            csv_dir_list.append("/".join(splitted_path[:len(splitted_path)-1]))
    
    csv_data = pd.DataFrame(columns=['f2_min', 'f2_max', 'f2_mean', 'f2_std', 'f3_min', 'f3_max', 'f3_mean', 'f3_std', 'f4_min', 'f4_max', 'f4_mean', 'f4_std', 'f5_min', 'f5_max', 'f5_mean', 'f5_std', 'f8_min', 'f8_max', 'f8_mean', 'f8_std', 'f10_min', 'f10_max', 'f10_mean', 'f11_num_peaks', 'f12_min', 'f12_max', 'f12_std', 'f13_min', 'f13_max', 'f13_std', 'f15_min', 'f15_std', 'f16_min', 'f16_std', 'f18_min', 'f18_std', 'f19_min', 'f19_max', 'f19_mean', 'f19_std', 'f20_min', 'f20_max', 'f20_mean', 'f20_std', 'f22_min', 'f22_max', 'f22_mean', 'f22_std', 'f23_min', 'f23_max', 'f23_mean', 'f23_std', 'f24_min', 'f24_max', 'f24_mean', 'f24_std', 'f25_mean', 'emotion'])
    for csv_dir in csv_dir_list:
        for file in os.listdir(csv_dir):
            if '.csv' in file:
                feature_extractor = CAERFeatureExtractor(os.path.join(csv_dir, file))
                match  csv_dir.rsplit("/",1)[1]:
                    case "Anger": 
                        label = 1
                    case "Disgust": 
                        label = 2
                    case "Fear": 
                        label = 3
                    case "Happy": 
                        label = 4
                    case "Sad": 
                        label = 5
                    case "Surprise": 
                        label = 6
                    case "Neutral": 
                        label = 7
                enhanced_dataframe = feature_extractor.get_element_list_as_dataframes()
                enhanced_dataframe['emotion'] = label
                dataframes_to_combine = csv_data, enhanced_dataframe
                csv_data = pd.concat(dataframes_to_combine, ignore_index=True)
                extracted_file_index += 1
                print(f"{extracted_file_index} of {len(csv_file_list)} csv files extracted.")
    dataframes_to_combine = combined_dataframe, csv_data
    combined_csv = pd.concat(dataframes_to_combine)
    dir_name = csv_directory.rsplit("/",1)[1]
    combined_csv.to_csv(csv_dir_list[0].rsplit("/",1)[0] + f"/extracted_{dir_name}_values.csv")
    print("Element extraction completed.")

def extract_all_csv_values():
    """
    Scan all directories of the CAER dataset and combine all csv files into one big file.
    
        Params:
            None
        Returns:
            None
    """
    csv_file_list = get_caer_csv_files(get_caer_directory())
    csv_dir_list = []
    for csv_file in csv_file_list:
        splitted_path = csv_file.split("/")
        if "/".join(splitted_path[:len(splitted_path)-1]) not in csv_dir_list:
            csv_dir_list.append("/".join(splitted_path[:len(splitted_path)-1]))
    print(f"Extracting Laban elements from {len(csv_file_list)} csv files in {len(csv_dir_list)} directories. Please wait...")
    dirs_to_extract = [csv_dir_list[0].rsplit("/",1)[0],csv_dir_list[10].rsplit("/",1)[0],csv_dir_list[20].rsplit("/",1)[0]]
    analyzed_directories = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        for result in executor.map(extract_values_in_dir, dirs_to_extract):
            if result != -1:
                analyzed_directories += 7
                print(f"{analyzed_directories} of {len(csv_dir_list)} directories have been analyzed.")

extract_all_csv_values()