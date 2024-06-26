import numpy as np
import pandas as pd

class FrameWindow:
    """
    This class contains laban element values calculated from pose data in every video frame 
    that is saved in its instance. 
    """
    def __init__(self):
        self.is_full = False
        self.f3_array = []
        self.f4_array = []
        self.f5_array = []
        self.f10_array = []
        self.f11_array = []
        self.f12_array = []
        self.f13_array = []
        self.f15_array = []
        self.f17_array = []
        self.f18_array = []
        self.f19_array = []
        self.f20_array = []
        self.f22_array = []
        self.f23_array = []
        self.f24_array = []
        self.f25_array = []
        self.z_movement_array = []
        self.frame_buffer = []
        self.elements_dataframe = ""
        self.emotion_array = []

    def calculate_elements_of_frame_buffer(self):
        """
        Use the values of every frame in this captured frame window to calculate the needed statistical values as mentioned in Aristidou et al. 2015 (see references folder).
        Params:
            None
        Returns:
            None
        """        
        for frame in self.frame_buffer:
            self.f3_array.append(frame.get_f3())
            self.f4_array.append(frame.get_f4())
            self.f5_array.append(frame.get_f5())
            self.f10_array.append(frame.get_f10())
            self.f11_array.append(frame.get_f11())
            self.f12_array.append(frame.get_f12())
            self.f13_array.append(frame.get_f13())
            self.f15_array.append(frame.get_f15())
            self.f17_array.append(frame.get_f17())
            self.f18_array.append(frame.get_f18())
            self.f19_array.append(frame.get_f19())
            self.f20_array.append(frame.get_f20())
            self.f22_array.append(frame.get_f22())
            self.f23_array.append(frame.get_f23())
            self.f24_array.append(frame.get_f24())
            self.f25_array.append(frame.get_f25())
            self.z_movement_array.append(frame.get_z_movement())
            self.emotion_array.append(frame.get_emotion())
        
        element_values = []
        element_columns = []
        f3_min = np.min(self.f3_array)
        element_values.append(f3_min)
        element_columns.append("f3_min")
        f3_max = np.max(self.f3_array)
        element_values.append(f3_max)
        element_columns.append("f3_max")
        f3_mean = np.mean(self.f3_array)
        element_values.append(f3_mean)
        element_columns.append("f3_mean")
        f3_std = np.std(self.f3_array)
        element_values.append(f3_std)
        element_columns.append("f3_std")
        f4_min = np.min(self.f4_array)
        element_values.append(f4_min)
        element_columns.append("f4_min")
        f4_max = np.max(self.f4_array)
        element_values.append(f4_max)
        element_columns.append("f4_max")
        f4_mean = np.mean(self.f4_array)
        element_values.append(f4_mean)
        element_columns.append("f4_mean")
        f4_std = np.std(self.f4_array)
        element_values.append(f4_std)
        element_columns.append("f4_std")
        f5_min = np.min(self.f5_array)
        element_values.append(f5_min)
        element_columns.append("f5_min")
        f5_max = np.max(self.f5_array)
        element_values.append(f5_max)
        element_columns.append("f5_max")
        f5_mean = np.mean(self.f5_array)
        element_values.append(f5_mean)
        element_columns.append("f5_mean")
        f5_std = np.std(self.f5_array)
        element_values.append(f5_std)
        element_columns.append("f5_std")
        f10_min = np.min(self.f10_array)
        element_values.append(f10_min)
        element_columns.append("f10_min")
        f10_max = np.max(self.f10_array)
        element_values.append(f10_max)
        element_columns.append("f10_max")
        f10_mean = np.mean(self.f10_array)
        element_values.append(f10_mean)
        element_columns.append("f10_mean")
        f11_num_peaks = 0
        f11_previous_sign = 1
        f11_sign = 1
        for accel_value in self.f11_array:
            if accel_value == self.f11_array[0]:
                if accel_value <= 0:
                    f11_previous_sign = -1
                else:
                    f11_previous_sign = 1
            else:
                if accel_value > 0:
                    f11_sign = 1
                else:
                    f11_sign = -1

                if f11_sign != f11_previous_sign:
                    f11_num_peaks += 1
                f11_previous_sign = f11_sign
        element_values.append(f11_num_peaks)
        element_columns.append("f11_num_peaks")
        f12_min = np.min(self.f12_array)
        element_values.append(f12_min)
        element_columns.append("f12_min")
        f12_max = np.max(self.f12_array)
        element_values.append(f12_max)
        element_columns.append("f12_max")
        f12_std = np.std(self.f12_array)
        element_values.append(f12_std)
        element_columns.append("f12_std")
        f13_min = np.min(self.f13_array)
        element_values.append(f13_min)
        element_columns.append("f13_min")
        f13_max = np.max(self.f13_array)
        element_values.append(f13_max)
        element_columns.append("f13_max")
        f13_std = np.std(self.f13_array)
        element_values.append(f13_std)
        element_columns.append("f13_std")
        f15_min = np.min(self.f15_array)
        element_values.append(f15_min)
        element_columns.append("f15_min")
        f15_std = np.std(self.f15_array)
        element_values.append(f15_std)
        element_columns.append("f15_std")
        f17_min = np.min(self.f17_array)
        element_values.append(f17_min)
        element_columns.append("f17_min")
        f17_std = np.std(self.f17_array)
        element_values.append(f17_std)
        element_columns.append("f17_std")
        f15_num_peaks = 0
        f15_previous_sign = 1
        f15_previous_value = 0
        f15_sign = 1
        for accel_value in self.f15_array:
            if accel_value == self.f15_array[0]:
                if accel_value <= 0:
                    f15_previous_sign = -1
                else:
                    f15_previous_sign = 1
                f15_previous_value = accel_value
            else:
                if accel_value > f15_previous_value:
                    f15_sign = 1
                else:
                    f15_sign = -1

                if f15_sign != f15_previous_sign:
                    f15_num_peaks += 1
                f15_previous_value = accel_value
                f15_previous_sign = f15_sign
        element_values.append(f15_num_peaks)
        element_columns.append("f18")
        f19_min = np.min(self.f19_array)
        element_values.append(f19_min)
        element_columns.append("f19_min")
        f19_max = np.max(self.f19_array)
        element_values.append(f19_max)
        element_columns.append("f19_max")
        f19_mean = np.mean(self.f19_array)
        element_values.append(f19_mean)
        element_columns.append("f19_mean")
        f19_std = np.std(self.f19_array)
        element_values.append(f19_std)
        element_columns.append("f19_std")
        f20_min = np.min(self.f20_array)
        element_values.append(f20_min)
        element_columns.append("f20_min")
        f20_max = np.max(self.f20_array)
        element_values.append(f20_max)
        element_columns.append("f20_max")
        f20_mean = np.mean(self.f20_array)
        element_values.append(f20_mean)
        element_columns.append("f20_mean")
        f20_std = np.std(self.f20_array)
        element_values.append(f20_std)
        element_columns.append("f20_std")
        f22_min = np.min(self.f22_array)
        element_values.append(f22_min)
        element_columns.append("f22_min")
        f22_max = np.max(self.f22_array)
        element_values.append(f22_max)
        element_columns.append("f22_max")
        f22_mean = np.mean(self.f22_array)
        element_values.append(f22_mean)
        element_columns.append("f22_mean")
        f22_std = np.std(self.f22_array)
        element_values.append(f22_std)
        element_columns.append("f22_std")
        f23_min = np.min(self.f23_array)
        element_values.append(f23_min)
        element_columns.append("f23_min")
        f23_max = np.max(self.f23_array)
        element_values.append(f23_max)
        element_columns.append("f23_max")
        f23_mean = np.mean(self.f23_array)
        element_values.append(f23_mean)
        element_columns.append("f23_mean")
        f23_std = np.std(self.f23_array)
        element_values.append(f23_std)
        element_columns.append("f23_std")
        f24_min = np.min(self.f24_array)
        element_values.append(f24_min)
        element_columns.append("f24_min")
        f24_max = np.max(self.f24_array)
        element_values.append(f24_max)
        element_columns.append("f24_max")
        f24_mean = np.mean(self.f24_array)
        element_values.append(f24_mean)
        element_columns.append("f24_mean")
        f24_std = np.std(self.f24_array)
        element_values.append(f24_std)
        element_columns.append("f24_std")
        f25_mean = np.mean(self.f25_array)
        element_values.append(f25_mean)
        element_columns.append("f25_mean")
        z_movement_mean = np.mean(self.z_movement_array)
        element_values.append(z_movement_mean)
        element_columns.append("z_mean")
        z_movement_sum = np.sum(self.z_movement_array)
        element_values.append(z_movement_sum)
        element_columns.append("z_sum")
        emotion = np.mean(self.emotion_array)
        element_values.append(emotion)
        element_columns.append("emotion")
        element_values = [element_values]
        self.elements_dataframe = pd.DataFrame(data=element_values,columns=element_columns)