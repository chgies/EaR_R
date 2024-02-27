import os
import pandas as pd
from datamanager_caer import get_caer_csv_files
from datamanager_caer import get_caer_directory
from caer_feature_extractor import CAERFeatureExtractor

csv_file_list = get_caer_csv_files(get_caer_directory())
#print(csv_file_list)
csv_dir_list = []
for csv_file in csv_file_list:
    splitted_path = csv_file.split("/")
    if "/".join(splitted_path[:len(splitted_path)-1]) not in csv_dir_list:
        csv_dir_list.append("/".join(splitted_path[:len(splitted_path)-1]))


csv_dir_list = csv_dir_list[:1]
print(f"Extracting Laban elements from {len(csv_file_list)} csv files in {len(csv_dir_list)} directories. Please wait...")

test_csv = pd.DataFrame(columns=['f2_min', 'f2_max', 'f2_mean', 'f2_std', 'f3_min', 'f3_max', 'f3_mean', 'f3_std', 'f4_min', 'f4_max', 'f4_mean', 'f4_std', 'f5_min', 'f5_max', 'f5_mean', 'f5_std', 'f8_min', 'f8_max', 'f8_mean', 'f8_std', 'f10_min', 'f10_max', 'f10_mean', 'f11_num_peaks', 'f12_min', 'f12_max', 'f12_std', 'f13_min', 'f13_max', 'f13_std', 'f15_min', 'f15_std', 'f16_min', 'f16_std', 'f18_min', 'f18_std', 'f19_min', 'f19_max', 'f19_mean', 'f19_std', 'f20_min', 'f20_max', 'f20_mean', 'f20_std', 'f22_min', 'f22_max', 'f22_mean', 'f22_std', 'f23_min', 'f23_max', 'f23_mean', 'f23_std', 'f24_min', 'f24_max', 'f24_mean', 'f24_std', 'f25_mean'])
train_csv = pd.DataFrame(columns=['f2_min', 'f2_max', 'f2_mean', 'f2_std', 'f3_min', 'f3_max', 'f3_mean', 'f3_std', 'f4_min', 'f4_max', 'f4_mean', 'f4_std', 'f5_min', 'f5_max', 'f5_mean', 'f5_std', 'f8_min', 'f8_max', 'f8_mean', 'f8_std', 'f10_min', 'f10_max', 'f10_mean', 'f11_num_peaks', 'f12_min', 'f12_max', 'f12_std', 'f13_min', 'f13_max', 'f13_std', 'f15_min', 'f15_std', 'f16_min', 'f16_std', 'f18_min', 'f18_std', 'f19_min', 'f19_max', 'f19_mean', 'f19_std', 'f20_min', 'f20_max', 'f20_mean', 'f20_std', 'f22_min', 'f22_max', 'f22_mean', 'f22_std', 'f23_min', 'f23_max', 'f23_mean', 'f23_std', 'f24_min', 'f24_max', 'f24_mean', 'f24_std', 'f25_mean'])
validation_csv = pd.DataFrame(columns=['f2_min', 'f2_max', 'f2_mean', 'f2_std', 'f3_min', 'f3_max', 'f3_mean', 'f3_std', 'f4_min', 'f4_max', 'f4_mean', 'f4_std', 'f5_min', 'f5_max', 'f5_mean', 'f5_std', 'f8_min', 'f8_max', 'f8_mean', 'f8_std', 'f10_min', 'f10_max', 'f10_mean', 'f11_num_peaks', 'f12_min', 'f12_max', 'f12_std', 'f13_min', 'f13_max', 'f13_std', 'f15_min', 'f15_std', 'f16_min', 'f16_std', 'f18_min', 'f18_std', 'f19_min', 'f19_max', 'f19_mean', 'f19_std', 'f20_min', 'f20_max', 'f20_mean', 'f20_std', 'f22_min', 'f22_max', 'f22_mean', 'f22_std', 'f23_min', 'f23_max', 'f23_mean', 'f23_std', 'f24_min', 'f24_max', 'f24_mean', 'f24_std', 'f25_mean'])
extracted_file_index = 0
for csv_dir in csv_dir_list:
    csv_data = pd.DataFrame(columns=['f2_min', 'f2_max', 'f2_mean', 'f2_std', 'f3_min', 'f3_max', 'f3_mean', 'f3_std', 'f4_min', 'f4_max', 'f4_mean', 'f4_std', 'f5_min', 'f5_max', 'f5_mean', 'f5_std', 'f8_min', 'f8_max', 'f8_mean', 'f8_std', 'f10_min', 'f10_max', 'f10_mean', 'f11_num_peaks', 'f12_min', 'f12_max', 'f12_std', 'f13_min', 'f13_max', 'f13_std', 'f15_min', 'f15_std', 'f16_min', 'f16_std', 'f18_min', 'f18_std', 'f19_min', 'f19_max', 'f19_mean', 'f19_std', 'f20_min', 'f20_max', 'f20_mean', 'f20_std', 'f22_min', 'f22_max', 'f22_mean', 'f22_std', 'f23_min', 'f23_max', 'f23_mean', 'f23_std', 'f24_min', 'f24_max', 'f24_mean', 'f24_std', 'f25_mean'])
    for file in os.listdir(csv_dir):
        if '.csv' in file:
            feature_extractor = CAERFeatureExtractor(os.path.join(csv_dir, file))
            dataframes_to_combine = csv_data, feature_extractor.get_element_list_as_dataframes()
            csv_data = pd.concat(dataframes_to_combine)
            extracted_file_index += 1
            print(f"{extracted_file_index} of {len(csv_file_list)} csv files extracted.")
    label = csv_dir.split("/")[len(csv_dir.split("/"))-1]
    print(label)
    csv_data['emotion'] = label
    if csv_dir.split("/")[len(csv_dir.split("/"))-2] == "test":
        test_combine_dataframes = test_csv, csv_data
        test_csv = pd.concat(test_combine_dataframes)
        test_csv.drop(test_csv.columns[test_csv.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    if csv_dir.split("/")[len(csv_dir.split("/"))-2] == "train":
        train_combine_dataframes = train_csv, csv_data
        train_csv = pd.concat(test_combine_dataframes)
    if csv_dir.split("/")[len(csv_dir.split("/"))-2] == "validation":
        validation_combine_dataframes = validation_csv, csv_data
        validation_csv = pd.concat(test_combine_dataframes)

test_csv.to_csv(csv_dir_list[0].rsplit("/",2)[0] + "/extracted_test_values.csv")
train_csv.to_csv(csv_dir_list[0].rsplit("/",2)[0] + "/extracted_train_values.csv")
validation_csv.to_csv(csv_dir_list[0].rsplit("/",2)[0] + "/extracted_validation_values.csv")

print("Element extraction completed.")
