"""
This class contains all the important motion features of a video frame
Instances of this class are handled by feature_extraction class
"""
from scipy.spatial import distance
from shapely.geometry import Polygon
class CAERFrameFeatures:

    def __init__(self, frame):
        self.frame = frame

        # Body components
        self.f3 = 0.0
        self.f4 = 0.0
        self.f5 = 0.0
        # Effort components
        self.f10 = (0.0, 0.0, 0.0)
        self.f11 = 0.0
        self.f12 = 0.0
        self.f13 = 0.0
        self.f15 = 0.0
        self.f17 = 0.0
        self.f18 = 0.0
        # Shape components
        self.f19 = 0.0
        self.f20 = 0.0
        self.f22 = 0.0
        self.f23 = 0.0
        self.f24 = 0.0
        self.f25 = 1
        # Auxiliary components
        self.centroid_position = (0.0, 0.0, 0.0)
        self.face_and_centroid_points_list = []
        self.avg_hip_velocity = 0.0
        self.pelvis_velocity = 0.0
        self.face_centroid_movement_correlation = (1,1,1)
        self.pelvis_position = (0.0, 0.0, 0.0)
        self.lhand_position = (0.0, 0.0, 0.0)
        self.rhand_position = (0.0, 0.0, 0.0)
        self.lhip_position = (0.0, 0.0, 0.0)
        self.rhip_position = (0.0, 0.0, 0.0)
        self.points_array = []
        self.z_movement = 0.0

    def calculate_z_movement(self, last_points_array):
        for point in self.points_array:
            for last_point in last_points_array:
                self.z_movement += point[2]/last_point[2]
        
    def load_dataframe_into_object(self, dataframe):
        """
        Loading data from a given Pands Dataframe into the local Laban elements of
        this frame. Does this for the basic elements which can be determined from
        this frame. Elements which need previous frames (for velocity, acceleration) 
        are loaded in later functions

        Params:
            dataframe (Pandas Dataframe): A dataframe with all Mediapipe coordinates for this frame

        Returns:
            None
        """
        raw_df_data = dataframe.to_dict(orient='records')
        self.frame = raw_df_data[0]['frame']
        for row in raw_df_data:
            new_point = (row['x'],row['y'],row['z'])
            self.points_array.append(new_point)
        
        ##### calculating necessary f points as mentioned in ./references/Feature_Tabellen.pdf
        self.lhand_position = self.points_array[20]
        self.rhand_position = self.points_array[19]
        # f3 is rhand to lhand distance
        self.f3 = distance.euclidean(self.lhand_position,self.rhand_position)
        # f4 is hands to head distance, avg of both sides
        self.f4 = (distance.euclidean(self.lhand_position,self.points_array[0]) + distance.euclidean(self.rhand_position,self.points_array[0]))/2
        # f5 is pelvis height, distance of pelvis to ground
        # use middle points of ankles as ground and hips as pelvis
        ankle_midpoint = (((self.points_array[28][0] + self.points_array[27][0])/2), ((self.points_array[28][1] + self.points_array[27][1])/2), ((self.points_array[28][2] + self.points_array[27][2])/2))
        self.lhip_position = self.points_array[24]
        self.rhip_position = self.points_array[23]
        self.pelvis_position = (((self.points_array[24][0] + self.points_array[23][0])/2), ((self.points_array[24][1] + self.points_array[23][1])/2), ((self.points_array[24][2] + self.points_array[23][2])/2))
        self.f5 = distance.euclidean(ankle_midpoint, self.pelvis_position)
        # calculate centroid position, here: center of trapezoid of shoulders and hips        
        root_point_sum_x = self.points_array[11][0] + self.points_array[12][0] + self.points_array[23][0] + self.points_array[24][0]
        root_point_sum_y = self.points_array[11][1] + self.points_array[12][1] + self.points_array[23][1] + self.points_array[24][1]
        root_point_sum_z = self.points_array[11][2] + self.points_array[12][2] + self.points_array[23][2] + self.points_array[24][2]
        self.centroid_position = (root_point_sum_x/4,root_point_sum_y/4,root_point_sum_z/4)

        self.face_and_centroid_points_list = self.points_array[:10]    
        self.face_and_centroid_points_list.append(self.centroid_position)
        # f19 is bounding volume of all joints
        full_body_volume_list = [self.points_array[0], self.pelvis_position]
        full_body_volume_list += self.points_array
        left_vol = self.calculate_area_of_body("left", full_body_volume_list)
        right_vol = self.calculate_area_of_body("right", full_body_volume_list)
        self.f19 = left_vol + right_vol

        upper_body_points_list = self.points_array[0], self.pelvis_position, points_array[12], points_array[11], points_array[14], points_array[13], points_array[20], points_array[19], points_array[24], points_array[23]
        # f22 is volume of left side
        self.f22 = self.calculate_area_of_body("left", upper_body_points_list)
        # f23 is volume of right side
        self.f23 = self.calculate_area_of_body("right", upper_body_points_list)
        # f20 is volume of upper body
        self.f20 = self.f22 + self.f23

        """
            ToDo: think about how volume can be used: can become smaller if left volume
            is bigger, bc. mediapipe point coordinates are often lower than 1.0
        """

        # f24 is distance head to root joint
        self.f24 = distance.euclidean(self.points_array[0], self.pelvis_position)
   
        # f25 is relation of hand's position to body
        horizontal_head_level = self.face_and_centroid_points_list[0][1]
        for point in self.face_and_centroid_points_list:
            if point[1] > horizontal_head_level:
                horizontal_head_level = point[1]
        if self.lhand_position[1] > horizontal_head_level or self.rhand_position[1] > horizontal_head_level: 
            self.f25 = 2
        elif self.lhand_position[1] > self.centroid_position[1] or self.rhand_position[1] > self.centroid_position[1]:
            self.f25 = 1
        else:
            self.f25 = 0

    def sort_joint_points_by_value(self, value_to_sort, list_of_point_tuples):
        """
        Sorts given points in list by value using BubbleSort algorithm

        Params:
            value_to_sort (String): which value should be used to sort. Can be "x","Y" or "z"
            points_list: List of body joint points with x,y,z coordinates
        
        Returns:
            points_list (list): the sorted list
        """

        if value_to_sort == "x":
            value_index = 0
        elif value_to_sort == "y":
            value_index = 1
        elif value_to_sort == "z":
            value_index = 2

        points_as_list = list(list_of_point_tuples)
        for round in range(len(points_as_list)-1,0,-1):
            for index in range(round):
                if points_as_list[index][value_index] > points_as_list[index+1][value_index]:
                    temp = points_as_list[index]
                    points_as_list[index] = points_as_list[index+1]
                    points_as_list[index+1] = temp

        return points_as_list

    def calculate_area_of_body(self, side_to_calculate, points_list):
        """
        Calculates the area of one side of the upper body. Uses the x and y values of body joint points found by  MediaPipe pose landmarker.
        
        Params:
            side (String): which side of the body is calculated, as seen from the camera. Can be "left" and "right".
            body_points (List of tuples): a list of found joint points. For a correct calculation, the list needs to include all upper body points left or right of the nose point and the pelvis, i.e. nose, shoulder(s), elbow(s), wrist(s), hip(s), pelvis.
                                        The first element of this list needs to be the nose (Point 0), se second the pelvis(middle of the two hip points)
        Returns:
            body_area (float): The calculated area of all given points.
        """

        sorted_x_axis_list = self.sort_joint_points_by_value("x", points_list)						
        sorted_y_axis_list = self.sort_joint_points_by_value("y", points_list)						
        unused_points_list = list(points_list[2:])
        polygon_path_forth = [points_list[0]]
        polygon_path_back = []		
        if side_to_calculate == 'left':
            range_start = sorted_x_axis_list.index(points_list[1])-1
            range_stop = -1
            range_step = -1
        else:
            range_start = sorted_x_axis_list.index(points_list[1])-1
            range_stop = len(sorted_x_axis_list)-1
            range_step = 1
        for x_index in range(range_start, range_stop, range_step):						
            if sorted_x_axis_list[x_index] in unused_points_list:
                # if point on the left of the current point is beneath the "nose" point, add it to the path 					
                if sorted_y_axis_list.index(sorted_x_axis_list[x_index]) < sorted_y_axis_list.index(points_list[0]):					
                    polygon_path_forth.append(sorted_x_axis_list[x_index])				
                    unused_points_list.remove(sorted_x_axis_list[x_index])				
                # if it is above the face point, remove all smaller former points till the next point above or the nose from the path					
                else:					
                    polygon_path_forth.append(sorted_x_axis_list[x_index])				
                    unused_points_list.remove(sorted_x_axis_list[x_index])				
                    for point in polygon_path_forth:				
                        if point != points_list[0]:			
                            if point != sorted_x_axis_list[x_index]:		
                                if sorted_y_axis_list.index(point) < sorted_y_axis_list.index(sorted_x_axis_list[x_index]):	
                                    polygon_path_forth.remove(point)
                                    unused_points_list.append(point)
                                else:
                                    break
                            else:		
                                break	
                                               
        # Walking to the upper left point finished, now wandering down to the right till we get to th pelvis						
        polygon_path_back.append(polygon_path_forth[len(polygon_path_forth)-1])						
        if side_to_calculate == 'left':
            range_start = 0
            range_stop = sorted_x_axis_list.index(points_list[1])
            range_step = 1
        else:
            range_start = len(sorted_x_axis_list)-1
            range_stop = sorted_x_axis_list.index(points_list[1])
            range_step = -1
        for x_index in range(range_start, range_stop, range_step):					
            if sorted_x_axis_list[x_index] in unused_points_list:                 					
                # if the current point is above the last, add it to the path.					
                if sorted_y_axis_list.index(sorted_x_axis_list[x_index]) < sorted_y_axis_list.index(polygon_path_back[0]):					
                    polygon_path_back.append(sorted_x_axis_list[x_index])		
                    unused_points_list.remove(sorted_x_axis_list[x_index])

                # if it is beneath , all former points that are above this point and not the first one, will be deleted							
                else:					
                    polygon_path_back.append(sorted_x_axis_list[x_index])				
                    unused_points_list.remove(sorted_x_axis_list[x_index])				
                                    
                    for point in polygon_path_back:				
                        if polygon_path_back.index(point) !=0:			
                            if point != sorted_x_axis_list[x_index]:
                                if point[1] > sorted_y_axis_list.index(sorted_y_axis_list[x_index]):	
                                    polygon_path_back.remove(point)
                                    unused_points_list.append(point)
                                else:
                                    break
                            else:		
                                break	

        polygon_path_back.append(points_list[1])

        # reached all points, now adding both paths together and create polygon						
        full_polygon_path = polygon_path_forth + polygon_path_back[1:]						
        polygon = Polygon(full_polygon_path)						
        volume = polygon.area						
        return volume

    def get_lhip_position(self):
        return self.lhip_position
    
    def get_rhip_position(self):
        return self.rhip_position
 
    def get_pelvis_and_hands_positions(self):
        """
        Return the pelvis and hands positions for this frame

        Params:
            None

        Returns:
            points_list (list of tuples): A list which consists of the palvis, left and right hand and hips position
        """
        points_list = [self.pelvis_position, self.rhand_position, self.lhand_position, self.lhip_position, self.rhip_position]
        return points_list
    
    def get_face_points_list(self):
        return self.face_and_centroid_points_list

    def calc_head_orientation(self, previous_points_list):
        """
        Calculate the correlation between head and body orientation to find "Effort:Space" Laban element. 
        Finds a tuple containing the ratio of face and centroid movement in x,y and z axis.
        Params:
            previous_points_list (list of tuples): A list containing the mediapipe face points and the centroid from last frame
        Returns:
            None
        """
        previous_face_list = previous_points_list[1:]
        #previous_face_list[:2]
        previous_centroid = previous_points_list[len(previous_points_list)-1]
        summed_x_change = 0
        summed_y_change = 0
        summed_z_change = 0        
        if self.frame > 0:
            for index in range(0,len(previous_face_list)-1):
                summed_x_change += (self.face_and_centroid_points_list[index][0] - previous_face_list[index][0])
                
                summed_y_change += (self.face_and_centroid_points_list[index][1] - previous_face_list[index][1])

                summed_z_change += (self.face_and_centroid_points_list[index][2] - previous_face_list[index][2])

        face_movement = (summed_x_change, summed_y_change, summed_z_change)

        centroid_x_movement = self.centroid_position[0] - previous_centroid[0]
        centroid_y_movement = self.centroid_position[1] - previous_centroid[1]
        centroid_z_movement = self.centroid_position[2] - previous_centroid[2]

        centroid_movement = [centroid_x_movement,centroid_y_movement, centroid_z_movement]
        if self.frame > 0:
            if centroid_movement[0] == 0.0:
                centroid_movement[0] = 1
            if centroid_movement[1] == 0.0:
                centroid_movement[1] = 1
            if centroid_movement[2] == 0.0:
                centroid_movement[2] = 1
            self.f10 = (face_movement[0]/centroid_movement[0], face_movement[1]/centroid_movement[1], face_movement[2]/centroid_movement[2])
        
    def calc_velocities(self, previous_frame_points_list, last_frame):
        """
        Calculate the velicities of the hips and hands
        Params:
            previous_frame_points_list (list of tuples): A list containing the left and right hips and hands position from the previuous frame
            last_frame: The number of the last frame
        Returns:
            None
        """
        if self.frame == 0:
            last_pelvis_position = self.pelvis_position
            last_rhand_position = self.rhand_position
            last_lhand_position = self.lhand_position
            last_lhip_position = self.lhip_position
            last_rhip_position = self.rhip_position
            seconds_from_last_frame = 1/30.0
        else:
            last_pelvis_position = previous_frame_points_list[0]
            last_rhand_position = previous_frame_points_list[1]
            last_lhand_position = previous_frame_points_list[2]
            last_lhip_position = previous_frame_points_list[3]
            last_rhip_position = previous_frame_points_list[4]
            seconds_from_last_frame = (self.frame - last_frame)/30.0
        
        self.f12 = distance.euclidean(last_pelvis_position, self.pelvis_position)
        self.pelvis_velocity = self.f12/seconds_from_last_frame 
        # f12 is velocity of hips over time period
        lhip_velocity = distance.euclidean(last_lhip_position, self.lhip_position)/seconds_from_last_frame
        rhip_velocity = distance.euclidean(last_rhip_position, self.rhip_position)/seconds_from_last_frame
        self.avg_hip_velocity = (lhip_velocity + rhip_velocity)/2
        # f13 is avg velocity of hands
        lhand_velocity = distance.euclidean(last_lhand_position, self.lhand_position)/seconds_from_last_frame
        rhand_velocity = distance.euclidean(last_rhand_position, self.rhand_position)/seconds_from_last_frame
        self.f13 = (lhand_velocity + rhand_velocity)/2
        
    def get_velocities(self):
        """
        Return the velocities of the hands, hips and pelvis
        Params:
            None
        Returns:
            velocities_list: A list containing the velocity of the hips and hands 
        """
        velocities_list = [self.avg_hip_velocity, self.f13, self.pelvis_velocity]
        return velocities_list

    def calc_accelerations(self, previous_frame_velocities, last_frame):
        """
        Calculate the acceleration of the pelvis
        Params:
            previous_velocities_list (list of tuples): A list containing the pelvis, left and right hands velocities from the previuous frame
            last_frame: The number of the last frame
        Returns:
            None
        """
        previous_hips_velocity = previous_frame_velocities[0]
        previous_hands_velocity = previous_frame_velocities[1]
        previous_pelvis_velocity = previous_frame_velocities[2]
        # f11 is deceleration of pelvis
        # f15 is acceleration of hands
        # f17 is acceleration of hips 
        if self.frame == 0:
            self.f11 = 0
            self.f15 = 0
            self.f17 = 0
        else:
            self.f11 = -(self.pelvis_velocity - previous_pelvis_velocity)/((self.frame - last_frame)/30.0)
            self.f15 = (self.f13 - previous_hands_velocity)/((self.frame - last_frame)/30.0)
            self.f17 = (self.avg_hip_velocity - previous_hips_velocity)/((self.frame - last_frame)/30.0)
        
    def get_accelerations(self):
        """
        Return the velocities of the pelvis and hands
        Params:
            None
        Returns:
            acceleration_list: A list of the pelvis and hands acceleration
        """
        acceleration_list = [self.f15]
        return acceleration_list

    def calc_jerk(self, previous_frames_acceleration_list, last_frame):
        """
        Calculate the rate of change in pelvis and hands acceleration ("Jerk")
        Params:
            previous_frame_acceleration_list (list of tuples): A list containing the pelvis, left and right hands acceleration from the previuous frame
            last_frame: The number of the last frame
        Returns:
            None
        """
        previous_hands_acceleration = previous_frames_acceleration_list[0]
        # f18 is derivative of f15 accelerations with respect to time ("Jerk") -> rate of changes in acceleration
        if self.frame > 0:
            self.f18 = (self.f15 - previous_hands_acceleration)/((self.frame - last_frame)/30.0)
        else:
            self.f18 = 0
        
    ### Getter and Setter methods
        
    def get_frame(self):
        return self.frame
    
    def set_frame(self, value):
        self.frame = value

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

    def get_f15(self):
        return self.f15

    def set_f15(self, value):
        self.f15 = value

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

    def get_mp_points(self):
        return self.points_array
    
    def get_z_movement(self):
        return self.z_movement