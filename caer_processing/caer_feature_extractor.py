import pandas as pd
from caer_frame_features import CAERFrameFeatures
from frame_window import FrameWindow
"""
This file contains all variables and functions to extract the features from 
caer video pose data. For more information, look into my thesis :)
"""

class CAERFeatureExtractor:
    """
    This class organizes the csv-file-to-laban-elements workflow. It reads a csv file with pose data of a video of the
    CAER dataset (captured with Mediapipe pose landmarker). This data get's converted into Laban element values as mentioned
    in Aristidou et al. 2015 (see reference directory). After that, the correspondent Laban components get calculated to
    make them usable for machine learning
    """
    def __init__(self, path_to_csv_file):
        """
        Constructor for this class.

        Params:
            path_to_scv_file (String): The path for the csv file which contains the pose data
        Returns:
            None
        """
        self.path_to_csv_file = path_to_csv_file
        self.frame_feature_array = []
        self.element_dataframes = []
        self.csv_data = self.load_csv_into_memory()
        self.convert_coords_to_laban()
        self.calc_laban_elements_of_video()

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

    def get_element_dataframes(self):
        """
        Returns the calculated Laban element dataframes for every frame window in this video

        Params:
            None
        Returns:
            element_dataframes (List): a list of Pandas Dataframes of every sliding window created
        """    
        return self.element_dataframes
    
    def calc_laban_elements_of_video(self):
        """
        Use the calculated laban elements saved in "self.frame_feature_array" variable to create
        a sliding window datastructure where the data for a specific amount of video frames is saved.
        This is used to calculate some statistical data for every element for further processing 
        
        Params:
            None
        Returns:
            None
        """
        window_size = 45
        frames_to_start_new_sliding_window = 5
        sliding_window_array = []
        max_index = len(self.frame_feature_array)
        sliding_window_array_is_full = False
        for frame_index in range(0, max_index):
            if frame_index % frames_to_start_new_sliding_window == 0:
                if not sliding_window_array_is_full:
                    rest_index = max_index-1-frame_index
                    if rest_index < window_size:
                        sliding_window_array_is_full = True
                    new_window = FrameWindow()
                    sliding_window_array.append(new_window)            
            for frame_window in sliding_window_array:
                if not frame_window.is_full:
                    frame_window.frame_buffer.append(self.frame_feature_array[frame_index])
                    if len(frame_window.frame_buffer) == window_size:
                        frame_window.is_full = True
                        frame_window.calculate_elements_of_frame_buffer()
                        self.element_dataframes.append(frame_window.elements_dataframe)
            """
            USE THIS CODE LATER IN OTHER FUNCTION WHEN A SLIDING WINDOW IS NEEDED

            frame_buffer.append(self.frame_feature_array[frame_index])
            if len(frame_buffer) == window_size:
                # Perform calculations on the current window of frames
                #print(len(frame_buffer))
                self.calculate_elements_of_frame_buffer(frame_buffer)
                #print("Processed Data:", processed_data)

                frame_buffer.pop(0)  # Slide the window by removing the oldest frame
            """
        
    def convert_coords_to_laban(self):
        """
        Converting the Mediapipe coordinates previously loaded into memory into Laban elements used to determine
        the Laban components

        Params:
            None
        Returns: 
            None
        """
        frame_size = int(self.csv_data['frame'].iloc[-1])
        for frame_index in range(0,frame_size +1):
            feature_object = CAERFrameFeatures(frame_index)
            feature_object.load_dataframe_into_object(self.csv_data.loc[self.csv_data['frame'] == frame_index])
            self.frame_feature_array.append(feature_object)
            if frame_index == 1:
                feature_object.calc_velocities([(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)], frame_index-1)
                feature_object.calc_accelerations([0,0,0], frame_index-1)
            else:
                last_points_list = self.frame_feature_array[frame_index-1].get_ph_positions()
                feature_object.calc_velocities(last_points_list, frame_index-1)
                previous_velocities_list = self.frame_feature_array[frame_index-1].get_velocities()
                feature_object.calc_accelerations(previous_velocities_list, frame_index-1)
                previous_accelerations_list = self.frame_feature_array[frame_index-1].get_accelerations()
                feature_object.calc_jerk(previous_accelerations_list, frame_index-1)
                previous_face_list = self.frame_feature_array[frame_index-1].get_face_points_list()
                feature_object.calc_head_body_angle(previous_face_list)


caer_feature_extractor = CAERFeatureExtractor("caer_processing/CAER_pose_example.csv")