import pandas as pd
import numpy as np
from caer_frame_features import CAERFrameFeatures

"""
This file contains all variables and functions to extract the features from 
caer video pose data. For more information, look into my thesis :)
"""

class CAERFeatureExtractor:
    def __init__(self, path_to_csv_file):
        self.path_to_csv_file = path_to_csv_file
        self.frame_feature_array = []
        self.csv_data = self.load_csv_into_memory()
        self.convert_coords_to_laban()
        print("created")

    def load_csv_into_memory(self):
        """
        Load a csv file and return it

        Params:
            None

        Returns:
            A Pandas Datagram of the loaded csv file
        """
        csv_data = pd.read_csv(self.path_to_csv_file)
        return csv_data
    
    def calc_laban_elements_of_video(self):
        """
        Calculate the Laban elements. ToDo
        """
        print("Calculating:")

        frame_buffer = []
        """
        ToDo: Create a frame window all 5 frames, not every Frame
        """
        window_size = 45

        sliding_window_array = []

        for frame_index in range(0, len(self.frame_feature_array)):
            
            
            frame_buffer.append(self.frame_feature_array[frame_index])

            if len(frame_buffer) == window_size:
                # Perform calculations on the current window of frames
                #print(len(frame_buffer))
                self.calculate_elements_of_frame_buffer(frame_buffer)
                #print("Processed Data:", processed_data)

                frame_buffer.pop(0)  # Slide the window by removing the oldest frame
            
        
    def calculate_elements_of_frame_buffer(self, frame_buffer):
        
        f2_array = []
        f3_array = []
        f4_array = []
        f5_array = []
        f8_array = []
        f10_array = []
        f11_array = []
        f12_array = []
        f13_array = []
        f15_array = []
        f16_array = []
        f18_array = []
        f19_array = []
        f20_array = []
        f21_array = []
        f22_array = []
        f23_array = []
        f24_array = []
        f25_array = []

        frame_index = 0
        
        for frame in frame_buffer:
            f2_array.append(frame.get_f2())
            f3_array.append(frame.get_f3())
            f4_array.append(frame.get_f4())
            f5_array.append(frame.get_f5())
            f8_array.append(frame.get_f8())
            f10_array.append(frame.get_f10())
            f11_array.append(frame.get_f11())
            f12_array.append(frame.get_f12())
            f13_array.append(frame.get_f13())
            f15_array.append(frame.get_f15())
            f16_array.append(frame.get_f16())
            f18_array.append(frame.get_f18())
            f19_array.append(frame.get_f19())
            f20_array.append(frame.get_f20())
            f22_array.append(frame.get_f22())
            f23_array.append(frame.get_f23())
            f24_array.append(frame.get_f24())
            f25_array.append(frame.get_f25())
            
            frame_index += 1
        
        f2_min = np.min(f2_array)
        f2_max = np.max(f2_array)
        f2_mean = np.mean(f2_array)
        f2_std = np.std(f2_array)
        f3_min = np.min(f3_array)
        f3_max = np.max(f3_array)
        f3_mean = np.mean(f3_array)
        f3_std = np.std(f3_array)
        f4_min = np.min(f4_array)
        f4_max = np.max(f4_array)
        f4_mean = np.mean(f4_array)
        f4_std = np.std(f4_array)
        f5_min = np.min(f5_array)
        f5_max = np.max(f5_array)
        f5_mean = np.mean(f5_array)
        f5_std = np.std(f5_array)
        f8_min = np.min(f8_array)
        f8_max = np.max(f8_array)
        f8_mean = np.mean(f8_array)
        f8_std = np.std(f8_array)
        f10_min = np.min(f10_array)
        f10_max = np.max(f10_array)
        f10_mean = np.mean(f10_array)
        f11_num_peaks = np.sum(f11_array)
        f12_min = np.min(f12_array)
        f12_max = np.max(f12_array)
        f12_std = np.std(f12_array)
        f13_min = np.min(f13_array)
        f13_max = np.max(f13_array)
        f13_std = np.std(f13_array)
        f15_min = np.min(f15_array)
        f15_std = np.std(f15_array)
        f16_min = np.min(f16_array)
        f16_std = np.std(f16_array)
        f18_min = np.min(f18_array)
        f18_std = np.std(f18_array)
        f19_min = np.min(f19_array)
        f19_max = np.max(f19_array)
        f19_mean = np.mean(f19_array)
        f19_std = np.std(f19_array)
        f20_min = np.min(f20_array)
        f20_max = np.max(f20_array)
        f20_mean = np.mean(f20_array)
        f20_std = np.std(f20_array)
        f21_min = np.min(f21_array)
        f21_max = np.max(f21_array)
        f21_mean = np.mean(f21_array)
        f21_std = np.std(f21_array)
        f22_min = np.min(f22_array)
        f22_max = np.max(f22_array)
        f22_mean = np.mean(f22_array)
        f22_std = np.std(f22_array)
        f23_min = np.min(f23_array)
        f23_max = np.max(f23_array)
        f23_mean = np.mean(f23_array)
        f23_std = np.std(f23_array)
        f24_min = np.min(f24_array)
        f24_max = np.max(f24_array)
        f24_mean = np.mean(f24_array)
        f24_std = np.std(f24_array)
        f_25_mean = np.mean(f25_array)
            
        #return np.mean(frames)
        
    def convert_coords_to_laban(self):
        """
        Converting the Mediapipe coordinates previously loaded into memory into Laban elements used to determine
        the Laban components

        Params:
            None
        Returns: 
            None
        """
        print("converting csv data into laban element values")
        frame_size = int(self.csv_data['frame'].iloc[-1])
        for frame_index in range(0,frame_size +1):
            feature_object = CAERFrameFeatures(frame_index)
            feature_object.load_dataframe_into_object(self.csv_data.loc[self.csv_data['frame'] == frame_index])
            self.frame_feature_array.append(feature_object)
            if frame_index == 1:
                feature_object.calc_velocities([(0,0,0),(0,0,0),(0,0,0)], frame_index-1)
                feature_object.calc_accelerations([0,0], frame_index-1)
            else:
                last_points_list = self.frame_feature_array[frame_index-1].get_ph_positions()
                feature_object.calc_velocities(last_points_list, frame_index-1)
                previous_velocities_list = self.frame_feature_array[frame_index-1].get_velocities()
                feature_object.calc_accelerations(previous_velocities_list, frame_index-1)
                previous_accelerations_list = self.frame_feature_array[frame_index-1].get_accelerations()
                feature_object.calc_jerk(previous_accelerations_list, frame_index-1)
                previous_face_list = self.frame_feature_array[frame_index-1].get_face_points_list()
                feature_object.calc_head_body_angle(previous_face_list)

        self.calc_laban_elements_of_video()

caer_feature_extractor = CAERFeatureExtractor("caer_processing/CAER_pose_example.csv")