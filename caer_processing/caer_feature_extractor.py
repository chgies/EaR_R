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
import pandas as pd
from caer_frame_features import CAER_Frame_Features

class CAER_Feature_Extractor:
    def __init__(self, path_to_csv_file):
        self.path_to_csv_file = path_to_csv_file
        self.frame_feature_array = []
        self.csv_data = self.load_csv_into_memory()
        self.convert_coords_to_laban()
        print("created")

    def load_csv_into_memory(self):
        csv_data = pd.read_csv(self.path_to_csv_file)
        return csv_data
    
    def calc_laban_elements(self):
        print("Calculating:")

    def convert_coords_to_laban(self):
        print("converting csv data into laban element values")
        frame_size = int(self.csv_data['frame'].iloc[-1])
        for frame_index in range(0,frame_size +1):
            #print(self.csv_data.loc[self.csv_data['frame'] == frame_index])
            feature_object = CAER_Frame_Features(frame_index)
            feature_object.load_dataframe(self.csv_data.loc[self.csv_data['frame'] == frame_index])
            self.frame_feature_array.append(feature_object)
        
        #self.calc_laban_elements()

caer_feature_extractor = CAER_Feature_Extractor("caer_processing/CAER_pose_example.csv")