import numpy as np
import pandas as pd

class FrameLabanWindow:
    """
    This class contains laban element values calculated  and formatted int high-level Laban Movement values as described by Melzer et al. (2019)
    from pose data in every video frame that is saved in it's instance. 
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
        self.head_level_array = []
        self.frame_buffer = []
        self.elements_dataframe = ""

        # laban elements
        self.jump = 0
        self.rhythmicity = 0
        self.spread = 0
        self.free_and_light = 0
        self.up_and_rise = 0
        self.rotation = 0
        self.passive_weight = 0
        self.arms_to_upper_body = 0
        self.sink = 0
        self.head_drop = 0
        self.retreat = 0
        self.condense_and_enclose = 0
        self.bind = 0
        self.twist_and_back = 0
        self.strong = 0
        self.sudden = 0
        self.advance = 0
        self.direct = 0
        self.hands_to_head = 0
        self.hands_above_head = 0
        self.body_shift_backing = 0
        self.head_shake = 0
        self.hands_to_body = 0
        self.orientation_change_lr = 0
        self.hands_to_head_backing = 0
        self.hands_up_backing = 0
    
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
            self.head_level_array.append(frame.get_head_level())
        
        element_values = []
        element_columns = []

        """
        Laban values for Happiness
        """
        frame_amount = len(self.frame_buffer)
        # if pelvis height of last frame is bigger than first, then jump
        f5_jump = self.f5_array[len(self.f5_array)-1]-self.f5_array[0]
        if f5_jump >= 0:
            self.jump = 1
            print("JUMP")
        else:
            self.jump = 0
        element_values.append(self.jump)
        element_columns.append("jump")

        # if values of jerk are less than 1/3 the same +/- amounts, then 
        # jerk (acc/decc changing) is high
        pos_f18 = np.sum(np.array(self.f18_array) >= 0, axis=0)
        neg_f18 = np.sum(np.array(self.f18_array) <= 0, axis=0)
        if pos_f18 >= neg_f18/3 or neg_f18 >= pos_f18/3:
            self.rhythmicity = 1
            print("RHYTHMICITY")
        else:
            self.rhythmicity = 0
        element_values.append(self.rhythmicity)
        element_columns.append("rhythmicity")
        
        # if last upper body volume is bigger than first, then spread
        f20_spread = self.f20_array[len(self.f20_array)-1]-self.f20_array[0]
        if f20_spread >= 0:
            self.spread = 1
            print("SPREAD")
        else:
            self.spread = 0
        element_values.append(self.spread)
        element_columns.append("spread")
        
        # if no motion deceleration peaks and low jerk, then free and light
        pos_f11 = np.sum(np.array(self.f11_array) >= 0, axis=0)
        neg_f11 = np.sum(np.array(self.f11_array) <= 0, axis=0)
        if pos_f11 > 1 and neg_f11 > 1:
            light_weight = False
        else:
            light_weight = True
        if pos_f18 <= neg_f18/3 or neg_f18 <= pos_f18/3:
            free = True
        else:
            free = False
        if light_weight and free:
            self.free_and_light = 1
            print("FREE AND LIGHT")
        else:
            self.free_and_light = 0        
        element_values.append(self.free_and_light)
        element_columns.append("free_and_light")
        
        # if torso and head distance is rising: up and rise
        f24_rise = self.f24_array[len(self.f24_array)-1]-self.f24_array[0]
        if f24_rise >= 0:
            self.up_and_rise = 1
            print("UP AND RISE")
        else:
            self.up_and_rise = 0
        element_values.append(self.up_and_rise)
        element_columns.append("up_and_rise")
        
        # if lift side volume shrinks and right side rises (or vice versa), then rotation
        left_side_change = self.f22_array[len(self.f22_array)-1]-self.f22_array[0]
        right_side_change = self.f23_array[len(self.f23_array)-1]-self.f23_array[0]
        if left_side_change < 0:
            left_shrink = True
        else:
            left_shrink = False
        if right_side_change < 0:
            right_shrink = True
        else:
            right_shrink = False
        if left_shrink != right_shrink:
            rotate = True
        else:
            rotate = False

        if rotate:
            self.rotation = 1
            print("ROTATION")
        else:
            self.rotation = 0
        element_values.append(self.rotation)
        element_columns.append("rotation")
        
        """
        Laban values for Sadness
        """

        # if hands move slowly and accelerate slowly or even decelearate -> passive (Effort-)weight
        # f13 border of 3.3 is taken by calculating the std value of f13 the whole dataset and using the mean of it
        f13_border = 3.37
        # f15 border of 126.25 is taken by calculating the std value of f15 the whole dataset and using the mean of it
        f15_border = 126.25
        if np.std(self.f15_array) <= f15_border or np.std(self.f13_array) <= f13_border:
            self.passive_weight = 1
            print("PASSIVE WEIGHT")
        else:
            self.passive_weight = 0
        element_values.append(self.passive_weight)
        element_columns.append("passive_weight")
        
        # if arms are above chest, "Arms to upper body" is true
        if np.mean(self.f25_array) >= 1:
            self.arms_to_upper_body = 1
            print("ARMS TO UPPER BODY")
        else:
            self.arms_to_upper_body = 0
        element_values.append(self.arms_to_upper_body)
        element_columns.append("arms_to_upper_body")

        # if distance head-torso is lowering, then sink. Use f24_rise value
        if f24_rise < 0:
            self.sink = 1
            print("SINK")
        else:
            self.sink = 0
        element_values.append(self.sink)
        element_columns.append("sink")
        
        head_distance = self.head_level_array[len(self.head_level_array)-1]-self.head_level_array[0]
        # root_movement_border value of 0.03 is taken by calculating the std value of f12 the whole dataset and using the mean of it
        root_movement_border = 0.03
        root_joint_distance = self.f12_array[len(self.f12_array)-1]-self.f12_array[0]
        if head_distance < 0 and root_joint_distance <= root_movement_border:
            self.head_drop = 1
            print("HEAD DROP")
        else:
            self.head_drop = 0
        element_values.append(self.head_drop)
        element_columns.append("head_drop")
        
        """
        Laban values for Fear:
        """
        
        # if body moves backwards and root joint moved more than usual, then retreat
        if root_movement_border and self.z_movement_array[len(self.z_movement_array)-1]-self.z_movement_array[0] > 0:
            backwards = True
        else:
            backwards = False
        if root_joint_distance > root_movement_border and backwards:
            self.retreat = 1
        else:
            self.retreat = 0
        element_values.append(self.retreat)
        element_columns.append("retreat")
        
        # if first upper body volume is bigger than first, then enclose
        if f20_spread < 0:
            self.condense_and_enclose = 1
        else:
            self.condense_and_enclose = 0
        element_values.append(self.condense_and_enclose)
        element_columns.append("condense_and_enclose")

        # if many peaks in hand acceleration (high jerk), then bind        
        if pos_f18 > neg_f18/3 or neg_f18 > pos_f18/3:
            self.bind = 1
        else:
            self.bind = 0
        element_values.append(self.bind)
        element_columns.append("bind")
        
        # if body rotates and moves back, then "twist and backing"
        # if angle is less than 0 once, then rotation
        if backwards and rotate:
            self.twist_and_back = 1
        else:
            self.twist_and_back = 0        
        element_values.append(self.twist_and_back)
        element_columns.append("twist_and_back")
        
        """
        Laban values for Anger:
        """
        
        # if movement is not light, then strong
        if not light_weight:
            self.strong = 1
        else:
            self.strong = 0
        element_values.append(self.strong)
        element_columns.append("strong")
        
        # if hand movement is fast, then sudden
        if np.std(self.f13_array) > f13_border:
            self.sudden = 1
        else:
            self.sudden = 0
        element_values.append(self.sudden)
        element_columns.append("sudden")
        
        # if body moves forwards, then advance
        if not backwards:
            self.advance = 1
        else:
            self.advance = 0
        element_values.append(self.advance)
        element_columns.append("advance")
        
        # if head and body angle is not opposite, then direct
        neg_f10 = np.sum(np.array(self.f10_array) <= 0, axis=0)
        if len(neg_f10) > 0:
            self.direct = 0
        else:
            self.direct = 1
        element_values.append(self.direct)
        element_columns.append("direct")
        
        """
        Laban values for Surprise:
        """
        # if hand moves to head, then Movement "hands to head"
        hand_level = self.f25_array[len(self.f25_array)-1]
        h_h_distance_change = self.f4_array[len(self.f4_array)-1]-self.f4_array[0]
        if h_h_distance_change < 0 and hand_level > 0:
            self.hands_to_head = 1
        else:
            self.hands_to_head = 0
        element_values.append(self.hands_to_head)
        element_columns.append("hands_to_head")
        
        # if hand moves over head, then Movement "hands over head"
        if h_h_distance_change < 0 and hand_level > 1:
            self.hands_over_head = 1
        else:
            self.hands_over_head = 0
        element_values.append(self.hands_above_head)
        element_columns.append("hands_above_head")
        
        # if body volume rises and move backwards, then "body shift and backing"
        body_volume_change = self.f20_array[len(self.f20_array)-1]-self.f20_array[0]
        if body_volume_change > 0:
            body_shift = True
        else:
            body_shift = False
        if body_shift and backwards:
            self.body_shift_backing = 1
        else:
            self.body_shift_backing = 0
        element_values.append(self.body_shift_backing)
        element_columns.append("body_shift_backing")
        
        # if head/body orientation angle changes multiple times from pos to neg, then head shaking
        pos_f10 = np.sum(np.array(self.f10_array) > 0, axis=0)
        if len(neg_f10) > frame_amount/3 and len(pos_f10) > frame_amount/3:
            self.head_shake = 1
        else:
            self.head_shake = 0
        element_values.append(self.head_shake)
        element_columns.append("head_shake")
        
        """
        Laban values for Disgust:
        """

        # if distance hand-head rises and hand is on body, then "hands to body"
        if h_h_distance_change > 0 and hand_level < 2:
            self.hands_to_body = 1
        else:
            self.hands_to_body = 0
        element_values.append(self.hands_to_body)
        element_columns.append("hands_to_body")
        
        # if body rotates and angle between head and body is mostly the same, then "orientation change to left/right"
        if rotate and len(neg_f10) == frame_amount/4:
            self.orientation_change_lr = 1
        else:
            self.orientation_change_lr = 0
        element_values.append(self.orientation_change_lr)
        element_columns.append("orientation_change_to_lr")
        
        #if hands go to head and body moves backwards
        if self.hands_to_head == 1 and backwards:
            self.hands_to_head_backing = 1
        else:
            self.hands_to_head_backing = 0
        element_values.append(self.hands_to_head_backing)
        element_columns.append("hands_to_head_backing")
        
        # if hand to head distance shrinks or hands go over head, then "hands up"
        # if hands go up and body moves backwards, then movement "hands up and backing"
        if (self.hands_to_head or self.hands_above_head) and backwards:
            self.hands_up_backing = 1
        else:
            self.hands_up_backing = 0 
        element_values.append(self.hands_up_backing)
        element_columns.append("hands_up_backing")
        element_values = [element_values]
        self.elements_dataframe = pd.DataFrame(data=element_values,columns=element_columns)