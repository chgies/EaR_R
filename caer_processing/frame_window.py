import numpy as np

class FrameWindow:
    def __init__(self):
        self.is_full = False
        self.f2_array = []
        self.f3_array = []
        self.f4_array = []
        self.f5_array = []
        self.f8_array = []
        self.f10_array = []
        self.f11_array = []
        self.f12_array = []
        self.f13_array = []
        self.f15_array = []
        self.f16_array = []
        self.f18_array = []
        self.f19_array = []
        self.f20_array = []
        self.f22_array = []
        self.f23_array = []
        self.f24_array = []
        self.f25_array = []
        self.frame_buffer = []
    
    def calculate_elements_of_frame_buffer(self):        
        for frame in self.frame_buffer:
            self.f2_array.append(frame.get_f2())
            self.f3_array.append(frame.get_f3())
            self.f4_array.append(frame.get_f4())
            self.f5_array.append(frame.get_f5())
            self.f8_array.append(frame.get_f8())
            self.f10_array.append(frame.get_f10())
            self.f11_array.append(frame.get_f11())
            self.f12_array.append(frame.get_f12())
            self.f13_array.append(frame.get_f13())
            self.f15_array.append(frame.get_f15())
            self.f16_array.append(frame.get_f16())
            self.f18_array.append(frame.get_f18())
            self.f19_array.append(frame.get_f19())
            self.f20_array.append(frame.get_f20())
            self.f22_array.append(frame.get_f22())
            self.f23_array.append(frame.get_f23())
            self.f24_array.append(frame.get_f24())
            self.f25_array.append(frame.get_f25())
        
        f2_min = np.min(self.f2_array)
        f2_max = np.max(self.f2_array)
        f2_mean = np.mean(self.f2_array)
        f2_std = np.std(self.f2_array)
        f3_min = np.min(self.f3_array)
        f3_max = np.max(self.f3_array)
        f3_mean = np.mean(self.f3_array)
        f3_std = np.std(self.f3_array)
        f4_min = np.min(self.f4_array)
        f4_max = np.max(self.f4_array)
        f4_mean = np.mean(self.f4_array)
        f4_std = np.std(self.f4_array)
        f5_min = np.min(self.f5_array)
        f5_max = np.max(self.f5_array)
        f5_mean = np.mean(self.f5_array)
        f5_std = np.std(self.f5_array)
        f8_min = np.min(self.f8_array)
        f8_max = np.max(self.f8_array)
        f8_mean = np.mean(self.f8_array)
        f8_std = np.std(self.f8_array)
        f10_min = np.min(self.f10_array)
        f10_max = np.max(self.f10_array)
        f10_mean = np.mean(self.f10_array)
        f11_num_peaks = np.sum(self.f11_array)
        f12_min = np.min(self.f12_array)
        f12_max = np.max(self.f12_array)
        f12_std = np.std(self.f12_array)
        f13_min = np.min(self.f13_array)
        f13_max = np.max(self.f13_array)
        f13_std = np.std(self.f13_array)
        f15_min = np.min(self.f15_array)
        f15_std = np.std(self.f15_array)
        f16_min = np.min(self.f16_array)
        f16_std = np.std(self.f16_array)
        f18_min = np.min(self.f18_array)
        f18_std = np.std(self.f18_array)
        f19_min = np.min(self.f19_array)
        f19_max = np.max(self.f19_array)
        f19_mean = np.mean(self.f19_array)
        f19_std = np.std(self.f19_array)
        f20_min = np.min(self.f20_array)
        f20_max = np.max(self.f20_array)
        f20_mean = np.mean(self.f20_array)
        f20_std = np.std(self.f20_array)
        f22_min = np.min(self.f22_array)
        f22_max = np.max(self.f22_array)
        f22_mean = np.mean(self.f22_array)
        f22_std = np.std(self.f22_array)
        f23_min = np.min(self.f23_array)
        f23_max = np.max(self.f23_array)
        f23_mean = np.mean(self.f23_array)
        f23_std = np.std(self.f23_array)
        f24_min = np.min(self.f24_array)
        f24_max = np.max(self.f24_array)
        f24_mean = np.mean(self.f24_array)
        f24_std = np.std(self.f24_array)
        f_25_mean = np.mean(self.f25_array)