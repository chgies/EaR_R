import pandas as pd
from caer_frame_features import CAERFrameFeatures

"""
This file contains all variables and functions to extract the features from 
caer video pose data. For more information, look into my thesis :)
"""

"""
Nächste Schritte:
    - Wie Laban-elemente nutzen?
        -> Man könnte 
                1. aus den dafür notwendigen Werten einen wert berechnen
                   und daraus ableiten, wie hoch die wahrscheinlichkeit ist, dass
                   in diesem frame labanelement da ist. 
                   Diese werte könnte man 
                2. als labels nutzen für ein NN, dass mit Koordinaten eines Frames und den
                   Labels trainiert wird
                3. Profit
                4. Mit diesem Netz wird dann ein 2. Netz trainiert, das mit Emotionen funktioniert? Dennis fragen
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
    
    def calc_laban_elements(self):
        """
        Calculate the Laban elements. ToDo
        """
        print("Calculating:")

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
        self.calc_laban_elements()

previous_face_list = self.frame_feature_array[frame_index-1].get_face_list()

feature_object.calc_head_orientation(previous_face_list)

caer_feature_extractor = CAERFeatureExtractor("caer_processing/CAER_pose_example.csv")