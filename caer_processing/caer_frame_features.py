"""
This class contains all the important motion features of a video frame
Instances of this class are handled by feature_extraction class
"""
import pandas as pd
from scipy.spatial import distance
from shapely.geometry import Polygon
from shapely import Point
class CAER_Frame_Features:

    def __init__(self, frame):
        self.frame = frame

        # Body components
        self.f1 = (0.0, 0.0, 0.0)
        self.f2 = (0.0, 0.0, 0.0)
        self.f3 = (0.0, 0.0, 0.0)
        self.f4 = (0.0, 0.0, 0.0)
        self.f5 = (0.0, 0.0, 0.0)
        self.f6 = (0.0, 0.0, 0.0)
        self.f7 = (0.0, 0.0, 0.0)
        self.f8 = (0.0, 0.0, 0.0)

        # Effort components
        self.f9 = (0.0, 0.0, 0.0)
        self.f10 = (0.0, 0.0, 0.0)
        self.f11 = (0.0, 0.0, 0.0)
        self.f12 = (0.0, 0.0, 0.0)
        self.f13 = (0.0, 0.0, 0.0)
        self.f14 = (0.0, 0.0, 0.0)
        self.f15 = (0.0, 0.0, 0.0)
        self.f16 = (0.0, 0.0, 0.0)
        self.f17 = (0.0, 0.0, 0.0)
        self.f18 = (0.0, 0.0, 0.0)

        # Shape components
        self.f19 = (0.0, 0.0, 0.0)
        self.f20 = (0.0, 0.0, 0.0)
        self.f21 = (0.0, 0.0, 0.0)
        self.f22 = (0.0, 0.0, 0.0)
        self.f23 = (0.0, 0.0, 0.0)
        self.f24 = (0.0, 0.0, 0.0)
        self.f25 = (0.0, 0.0, 0.0)

        # Space components
        self.f26 = (0.0, 0.0, 0.0)
        self.f27 = (0.0, 0.0, 0.0)

        # laban elements
        self.jump = 0.0
        self.rhythmicity = 0.0
        self.spread = 0.0
        self.free_and_light = 0.0
        self.up_and_rise = 0.0
        self.rotation = 0.0
        self.passive_weight = 0.0
        self.arms_to_upper_body = 0.0
        self.sink = 0.0
        self.head_drop = 0.0
        self.retreat = 0.0
        self.condense_and_enclose = 0.0
        self.bind = 0.0
        self.twist_and_back = 0.0
        self.strong = 0.0
        self.sudden = 0.0
        self.advance = 0.0
        self.direct = 0.0
        self.hands_to_head = 0.0
        self.hands_above_head = 0.0
        self.body_shift_backing = 0.0
        self.head_shake = 0.0
        self.hands_to_body = 0.0
        self.orientation_change_lr = 0.0
        self.hands_to_head_backing = 0.0
        self.hands_up_backing = 0.0

    def load_dataframe(self, dataframe):
        raw_df_data = dataframe.to_dict(orient='records')
        self.frame = raw_df_data[0]['frame']
        points_array = []
        for row in raw_df_data:
            new_point = (row['x'],row['y'],row['z'])
            points_array.append(new_point)
       # print(points_array)
        
        ##### calculating necessary f points as mentioned in ./references/Feature_Tabellen.pdf

        # f1 is feet to hips distance, avg of both sides
        self.f1 = (distance.euclidean(points_array[28],points_array[24]) + distance.euclidean(points_array[27],points_array[23]))/2
        # f3 is rhand to lhand distance
        self.f3 = distance.euclidean(points_array[20],points_array[19])
        # f4 is hands to head distance, avg of both sides
        self.f4 = (distance.euclidean(points_array[20],points_array[0]) + distance.euclidean(points_array[19],points_array[0]))/2
        # f5 is pelvis height, distance of pelvis to ground
        # use middle points of ankles as ground and hips as pelvis
        ankle_midpoint = (((points_array[28][0]+points_array[28][0])/2), ((points_array[28][1]+points_array[28][1])/2), ((points_array[28][2]+points_array[28][2])/2))
        pelvis_position = (((points_array[24][0]+points_array[23][0])/2), ((points_array[24][1]+points_array[23][1])/2), ((points_array[24][2]+points_array[23][2])/2))
        self.f5 = distance.euclidean(ankle_midpoint, pelvis_position)
        # f10 is angle between head orientation and body path (trajectory of pelvis)
        self.f10 = 
        -> Head orientation messen: ob Abstand Auge-Nase l und r sich ändert?
        # f11 is deceleration of pelvis
        -> needs former frame position of pelvis
        self.f11 = 
        # f12 is distance of pelvis over time period
        -> needs former frame position of pelvis
        self.f12 = 
        # f13 is avg velocity of hands
        -> needs former frame position of hands
        self.f13 = 
        # f15 is derivative of hands velocities with respect to time
        -> solve f13 first
        self.f15 = 
        # f18 is derivative of f15 velocities with respect to time
        -> solve f15 first
        self.f18 = 
        # f19 is bounding volume of all joints
        self.f19 = 
        Punkte: 0 (Gesicht)
        - Wenn Hände weiter außen als schultern:
            - wenn hand über schulter, dann 0 - 16/15
            - sonst 0-12-16/0-11-15
        - sonst:
            - wenn hand über schulter, dann 0 - 16/15
            - sonst 0-12/0-16
        -> max x max y minx miny punkte, 
        -> polygon aus diesen Punkten
        -> prüfen ob andere Punkte in diesem Polygon: Wenn nein, dann mit einschließen?
        -> shapely.Polygon.area
        # f20 is volume of upper body
        self.f20 = 
        -> selbe wie f19
        # f22 is volume of left side
        self.f22 = 
        -> ähnlich f19
        # f23 is volume of right side
        self.f23 = 
        -> ähnlich f19
        # f24 is distance head to root joint
        pelvis_position = (((points_array[24][0]+points_array[23][0])/2), ((points_array[24][1]+points_array[23][1])/2), ((points_array[24][2]+points_array[23][2])/2))
        self.f24 = distance(pelvis_position, points_array[0])
   
        # f25 is relation of hand's position to body       
        self.f25 = 

        print(f"f1 on frame {self.frame}: {self.f1}")
        print(f"f3 on frame {self.frame}: {self.f3}")
        print(f"f4 on frame {self.frame}: {self.f4}")
        print(f"f5 on frame {self.frame}: {self.f5}")
        """
        print(f"f10 on frame {self.frame}: {self.f10}")
        print(f"f11 on frame {self.frame}: {self.f11}")
        print(f"f12 on frame {self.frame}: {self.f12}")
        print(f"f13 on frame {self.frame}: {self.f13}")
        print(f"f15 on frame {self.frame}: {self.f15}")
        print(f"f18 on frame {self.frame}: {self.f18}")
        print(f"f19 on frame {self.frame}: {self.f19}")
        print(f"f20 on frame {self.frame}: {self.f20}")
        print(f"f22 on frame {self.frame}: {self.f22}")
        print(f"f23 on frame {self.frame}: {self.f23}")
        print(f"f24 on frame {self.frame}: {self.f24}")
        print(f"f25 on frame {self.frame}: {self.f25}")
        """

    ### Getter and Setter methods
        
    def get_frame(self):
        return self.frame
    
    def set_frame(self, value):
        self.frame = value

    def get_f1(self):
        return self.f1

    def set_f1(self, value):
        self.f1 = value

    def get_f2(self):
        return self.f2

    def set_f2(self, value):
        self.f2 = value

    def get_f3(self):
        return self.f3

    def set_f3(self, value):
        self.f3 = value

    def get_f4(self):
        return self.f4

    def set_f4(self, value):
        self.f4 = value

    def get_f5(self):
        return self.f5

    def set_f5(self, value):
        self.f5 = value

    def get_f6(self):
        return self.f6

    def set_f6(self, value):
        self.f6 = value

    def get_f7(self):
        return self.f7

    def set_f7(self, value):
        self.f7 = value

    def get_f8(self):
        return self.f8

    def set_f8(self, value):
        self.f8 = value

    def get_f9(self):
        return self.f9

    def set_f9(self, value):
        self.f9 = value

    def get_f10(self):
        return self.f10

    def set_f10(self, value):
        self.f10 = value

    def get_f11(self):
        return self.f11

    def set_f11(self, value):
        self.f11 = value

    def get_f12(self):
        return self.f12

    def set_f12(self, value):
        self.f12 = value

    def get_f13(self):
        return self.f13

    def set_f13(self, value):
        self.f13 = value

    def get_f14(self):
        return self.f14

    def set_f14(self, value):
        self.f14 = value

    def get_f15(self):
        return self.f15

    def set_f15(self, value):
        self.f15 = value

    def get_f16(self):
        return self.f16

    def set_f16(self, value):
        self.f16 = value

    def get_f17(self):
        return self.f17

    def set_f17(self, value):
        self.f17 = value

    def get_f18(self):
        return self.f18

    def set_f18(self, value):
        self.f18 = value

    def get_f19(self):
        return self.f19

    def set_f19(self, value):
        self.f19 = value

    def get_f20(self):
        return self.f20

    def set_f20(self, value):
        self.f20 = value

    def get_f21(self):
        return self.f21

    def set_f21(self, value):
        self.f21 = value

    def get_f22(self):
        return self.f22

    def set_f22(self, value):
        self.f22 = value

    def get_f23(self):
        return self.f23

    def set_f23(self, value):
        self.f23 = value

    def get_f24(self):
        return self.f24

    def set_f24(self, value):
        self.f24 = value

    def get_f25(self):
        return self.f25

    def set_f25(self, value):
        self.f25 = value

    def get_f26(self):
        return self.f26

    def set_f26(self, value):
        self.f26 = value

    def get_f27(self):
        return self.f27

    def set_f27(self, value):
        self.f27 = value

    def get_jump(self):
        return self.jump

    def set_jump(self, value):
        self.jump = value

    def get_rhythmicity(self):
        return self.rhythmicity

    def set_rhythmicity(self, value):
        self.rhythmicity = value

    def get_spread(self):
        return self.spread

    def set_spread(self, value):
        self.spread = value

    def get_free_and_light(self):
        return self.free_and_light

    def set_free_and_light(self, value):
        self.free_and_light = value

    def get_up_and_rise(self):
        return self.up_and_rise

    def set_up_and_rise(self, value):
        self.up_and_rise = value

    def get_rotation(self):
        return self.rotation

    def set_rotation(self, value):
        self.rotation = value

    def get_passive_weight(self):
        return self.passive_weight

    def set_passive_weight(self, value):
        self.passive_weight = value

    def get_arms_to_upper_body(self):
        return self.arms_to_upper_body

    def set_arms_to_upper_body(self, value):
        self.arms_to_upper_body = value

    def get_sink(self):
        return self.sink

    def set_sink(self, value):
        self.sink = value

    def get_head_drop(self):
        return self.head_drop

    def set_head_drop(self, value):
        self.head_drop = value

    def get_retreat(self):
        return self.retreat

    def set_retreat(self, value):
        self.retreat = value

    def get_condense_and_enclose(self):
        return self.condense_and_enclose

    def set_condense_and_enclose(self, value):
        self.condense_and_enclose = value

    def get_bind(self):
        return self.bind

    def set_bind(self, value):
        self.bind = value

    def get_twist_and_back(self):
        return self.twist_and_back

    def set_twist_and_back(self, value):
        self.twist_and_back = value

    def get_strong(self):
        return self.strong

    def set_strong(self, value):
        self.strong = value

    def get_sudden(self):
        return self.sudden

    def set_sudden(self, value):
        self.sudden = value

    def get_advance(self):
        return self.advance

    def set_advance(self, value):
        self.advance = value

    def get_direct(self):
        return self.direct

    def set_direct(self, value):
        self.direct = value

    def get_hands_to_head(self):
        return self.hands_to_head

    def set_hands_to_head(self, value):
        self.hands_to_head = value

    def get_hands_above_head(self):
        return self.hands_above_head

    def set_hands_above_head(self, value):
        self.hands_above_head = value

    def get_body_shift_backing(self):
        return self.body_shift_backing

    def set_body_shift_backing(self, value):
        self.body_shift_backing = value

    def get_head_shake(self):
        return self.head_shake

    def set_head_shake(self, value):
        self.head_shake = value

    def get_hands_to_body(self):
        return self.hands_to_body

    def set_hands_to_body(self, value):
        self.hands_to_body = value

    def get_orientation_change_lr(self):
        return self.orientation_change_lr

    def set_orientation_change_lr(self, value):
        self.orientation_change_lr = value

    def get_hands_to_head_backing(self):
        return self.hands_to_head_backing

    def set_hands_to_head_backing(self, value):
        self.hands_to_head_backing = value

    def get_hands_up_backing(self):
        return self.hands_up_backing

    def set_hands_up_backing(self, value):
        self.hands_up_backing = value