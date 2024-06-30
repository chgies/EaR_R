import warnings
import pandas as pd
from candor_processing.candor_frame_features import CANDORFrameFeatures
from candor_processing.frame_window import FrameWindow
from candor_processing.frame_laban_window import FrameLabanWindow
warnings.simplefilter(action='ignore', category=FutureWarning)

class CANDORFeatureExtractor:
    """
    This class organizes the csv-file-to-laban-elements workflow. It reads a csv file with pose data of a video of the
    CANDOR dataset (captured with Mediapipe pose landmarker). This data iss converted into Laban element values as mentioned
    in Aristidou et al. 2015 (see reference directory). After that, the correspondent Laban components get calculated to
    make them usable for machine learning
    """
    def __init__(self, input_data, use_laban_features, arg_is_file = True):
        """
        Constructor for this class.

        Params:
            path_to_scv_file (String): The path for the csv file which contains the pose data
        Returns:
            None
        """
        self.path_to_csv_file = input_data
        self.frame_feature_array = []
        self.element_dataframes = pd.DataFrame(columns=['f3_min', 'f3_max', 'f3_mean', 'f3_std', 'f4_min', 'f4_max', 'f4_mean', 'f4_std', 'f5_min', 'f5_max', 'f5_mean', 'f5_std', 'f10_min', 'f10_max', 'f10_mean', 'f11_num_peaks', 'f12_min', 'f12_max', 'f12_std', 'f13_min', 'f13_max', 'f13_std', 'f15_min', 'f15_std', 'f17_min', 'f17_std', 'f18', 'f19_min', 'f19_max', 'f19_mean', 'f19_std', 'f20_min', 'f20_max', 'f20_mean', 'f20_std', 'f22_min', 'f22_max', 'f22_mean', 'f22_std', 'f23_min', 'f23_max', 'f23_mean', 'f23_std', 'f24_min', 'f24_max', 'f24_mean', 'f24_std', 'f25_mean', 'z_mean', 'z_sum', 'emotion'])
        self.laban_dataframes = pd.DataFrame(columns=['jump', 'rhythmicity', 'spread', 'free_and_light', 'up_and_rise', 'rotation', 'passive_weight', 'arms_to_upper_body', 'sink', 'head_drop', 'retreat', 'condense_and_enclose', 'bind', 'twist_and_back', 'strong', 'sudden', 'advance', 'direct', 'hands_to_head', 'hands_above_head', 'body_shift_backing', 'head_shake', 'hands_to_body', 'orientation_change_to_lr', 'hands_to_head_backing', 'hands_up_backing'])
        
        if arg_is_file:
            self.csv_data = self.load_csv_into_memory()
        else:
            self.csv_data = input_data

        # convert coords to features and append to frame_feature_array        
        self.convert_raw_coords_to_features()
            
        # create sliding windows with features and append them to element_dataframes data
        if not use_laban_features:
            if arg_is_file:
                self.calc_feature_elements_of_video()
            else:
                self.calc_feature_elements_of_livestream_frame_buffer()
        # create sliding windows with Laban Motor Elements and append them to laban_dataframes data
        else:
            if arg_is_file:
                self.calc_laban_movement_of_video()
            else:
                self.calc_laban_movement_of_livestream_frame_buffer()

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

    def get_element_list_as_dataframes(self):
        """
        Returns the calculated Laban element dataframes for every frame window in this video

        Params:
            None
        Returns:
            element_dataframes (List): a list of Pandas Dataframes of every sliding window created
        """    
        return self.element_dataframes

# Feature Calculation following Aristidou
    def calc_feature_elements_of_livestream_frame_buffer(self):
        """
        Build a frame window out of give data from a livestream.
        Use the calculated laban elements saved in "self.frame_feature_array" variable to create
        a sliding window where the data for a specific amount of video frames is saved.
        This is used to calculate some statistical data for every element for further processing 

            Params:
                None
            Returns:
                None

        """

        frame_window = FrameWindow()
        max_index = len(self.frame_feature_array)
        for frame_index in range(0, max_index):
            frame_window.frame_buffer.append(self.frame_feature_array[frame_index])
            frame_window.calculate_elements_of_frame_buffer()
            dataframes_to_combine = self.element_dataframes, frame_window.elements_dataframe, self.frame_feature_array
            self.element_dataframes = pd.concat(dataframes_to_combine)

    def calc_feature_elements_of_video(self):
        """
        Build multiple frame windows out of a video file.
        Use the calculated laban elements saved in "self.frame_feature_array" variable to create
        a sliding window datastructure where the data for a specific amount of video frames is saved.
        This is used to calculate some statistical data for every element for further processing 
        
            Params:
                None
            Returns:
                None
        """
        if not self.csv_data.empty:
            window_size = 30
            frames_to_start_new_sliding_window = 30
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
                            dataframes_to_combine = self.element_dataframes, frame_window.elements_dataframe
                            self.element_dataframes = pd.concat(dataframes_to_combine)
                            

    def convert_raw_coords_to_features(self):
        """
        Converting the Mediapipe coordinates previously loaded into memory into hig level features used to determine
        the Laban components

            Params:
                None
            Returns: 
                None
        """
        if not self.csv_data.empty:    
            frame_size = int(self.csv_data['frame'].iloc[-1])
            empty_frames = 0
            for frame_index in range(0,frame_size):
                if self.csv_data.loc[self.csv_data['frame'] == frame_index].size > 0:
                    feature_object = CANDORFrameFeatures(frame_index)
                    feature_object.load_dataframe_into_object(self.csv_data.loc[self.csv_data['frame'] == frame_index])
                    self.frame_feature_array.append(feature_object)
                    if frame_index == 1:
                        feature_object.calc_velocities([(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)], frame_index-1)
                        feature_object.calc_accelerations([0,0,0], frame_index-1)
                    elif frame_index > 1 and frame_index <= len(self.frame_feature_array)-empty_frames:
                        for previous_empty_frames in range(1, len(self.frame_feature_array)-1):
                            if self.frame_feature_array[frame_index-previous_empty_frames] != 0:
                                last_points_list = self.frame_feature_array[frame_index-previous_empty_frames].get_pelvis_and_hands_positions()
                                feature_object.calc_velocities(last_points_list, frame_index-1)
                                feature_object.calculate_z_movement(self.frame_feature_array[frame_index-1].get_mp_points())
                                previous_velocities_list = self.frame_feature_array[frame_index-previous_empty_frames].get_velocities()
                                feature_object.calc_accelerations(previous_velocities_list, frame_index-1)
                                previous_accelerations_list = self.frame_feature_array[frame_index-1-previous_empty_frames].get_accelerations()
                                previous_hips_acceleration_value = self.frame_feature_array[frame_index-1-previous_empty_frames].get_hips_acceleration_value()
                                feature_object.calc_jerk(previous_accelerations_list, previous_hips_acceleration_value)
                                previous_face_list = self.frame_feature_array[frame_index-previous_empty_frames].get_face_points_list()
                                feature_object.calc_head_orientation(previous_face_list)
                                break
                else:
                    empty_frames += 1

    def calc_laban_movement_of_livestream_frame_buffer(self):
        """
        Build a frame window out of give data from a livestream.
        Use the calculated laban elements saved in "self.frame_feature_array" variable to create
        a sliding window where the data for a specific amount of video frames is saved.
        This is used to calculate some statistical data for every element for further processing 

            Params:
                None
            Returns:
                None

        """
        frame_laban_window = FrameLabanWindow()
        max_index = len(self.frame_feature_array)
        for frame_index in range(0, max_index):
            frame_laban_window.frame_buffer.append(self.frame_feature_array[frame_index])
            frame_laban_window.calculate_elements_of_frame_buffer()
            dataframes_to_combine = self.laban_dataframes, frame_laban_window.elements_dataframe
            self.laban_dataframes = pd.concat(dataframes_to_combine)

    # Feature Calculation to Laban Motor Elements
    def calc_laban_movement_of_video(self):
        """
        Build multiple frame windows out of a video file.
        Use the calculated laban elements saved in "self.frame_feature_array" variable to create
        a sliding window datastructure where the data for a specific amount of video frames is saved.
        This is used to calculate some statistical data for every element for further processing 
        
            Params:
                None
            Returns:
                None
        """
        if not self.csv_data.empty:
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
                        new_laban_window = FrameLabanWindow()
                        sliding_window_array.append(new_laban_window)            
                for frame_laban_window in sliding_window_array:
                    if not frame_laban_window.is_full:
                        frame_laban_window.frame_buffer.append(self.frame_feature_array[frame_index])
                        if len(frame_laban_window.frame_buffer) == window_size:
                            frame_laban_window.is_full = True
                            frame_laban_window.calculate_elements_of_frame_buffer()
                            dataframes_to_combine = self.laban_dataframes, frame_laban_window.elements_dataframe
                            self.laban_dataframes = pd.concat(dataframes_to_combine)

    def get_laban_movement_list_as_dataframes(self):
            """
            Returns the calculated Laban element dataframes for every frame window in this video

            Params:
                None
            Returns:
                element_dataframes (List): a list of Pandas Dataframes of every sliding window created
            """    
            return self.laban_dataframes