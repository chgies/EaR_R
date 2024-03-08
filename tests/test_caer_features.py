import unittest
from caer_processing import caer_frame_features

class MyTestCase(unittest.TestCase):

    def setUp(self):
        point_a = 1,1,1
        point_b = 3,-3,4
        point_c = 0.5, 7,7
        point_d = -4,3,-4
        point_e = 4,1,-2
        self.test_points = [point_a, point_b, point_c, point_d, point_e]
        
        self.feature_object = caer_frame_features.CAERFrameFeatures(0)

    def test_point_sorting(self):
        tested_x_axis_list = self.feature_object.sort_joint_points_by_value("x", self.test_points)
        tested_y_axis_list = self.feature_object.sort_joint_points_by_value("y", self.test_points)
        tested_z_axis_list = self.feature_object.sort_joint_points_by_value("z", self.test_points)
        self.assertEqual(tested_x_axis_list, [(-4,3,-4), (0.5,7,7), (1,1,1), (3,-3,4), (4,1,-2)])
        self.assertEqual(tested_y_axis_list, [(3,-3,4), (1,1,1), (4,1,-2), (-4,3,-4), (0.5,7,7)])
        self.assertEqual(tested_z_axis_list, [(-4,3,-4), (4,1,-2), (1,1,1), (3,-3,4), (0.5,7,7)])

    def test_volume_calculation(self):
        
        left_volume = self.feature_object.calculate_area_of_upper_body("left", self.test_points)
        right_volume = self.feature_object.calculate_area_of_upper_body("right", self.test_points)
        self.assertEqual(left_volume,22.5)
        self.assertEqual(right_volume, 6.0)

    pass