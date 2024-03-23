import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import time
import os
from urllib.request import urlretrieve

class CAERPoseAnalyzer():

    def __init__(self, video_path, landmark_type):
        self.output_window = None    # Used as screen variable to draw landmarks on
        self.last_timestamp_ms = 0   # needed by Mediapipe when in "LIVE_STREAM"-RunningMode
        
        self.analyzed_results_person = "frame,person,x,y,z\n"  # used to store pose data of the person with the id 0 in a conversation
        
        self.landmark_type = landmark_type
        self.landmark_options = self.create_landmark_options()
        self.video_path = video_path
        self.analyze_video()
        
    def check_pose_landmark_file(self):
        """
        This function checks if the needed Mediapipe Pose estimation landmark files are available in the sub directory
        '/landmark_files'. If not, they get downloaded from Mediapipe servers. 
        More info on https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

        Parameters:
        landmark_type (String): which of the current 3 models do you need for pose estimation. Can be 'lite', 'full' or 'heavy'

        Returns:
        The path to the file in string format
        """

        task_file_dir = './landmark_files/'
        match self.landmark_type:
            case 'lite':
                if os.path.isfile(task_file_dir + 'pose_landmarker_lite.task'):
                    return task_file_dir + 'pose_landmarker_lite.task'
                else:
                    print("Mediapipe Pose Landmark file 'lite' not yet downloaded. Downloading now into './landmark_files' sub directory.")
                    try:
                        url = (
                            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
                        )
                        filename = task_file_dir + "pose_landmarker_lite.task"
                        urlretrieve(url, filename)
                        print("Downloading Mediapipe Pose Landmark file 'pose_landmarker_lite.task' successful")
                        return task_file_dir + 'pose_landmarker_lite.task'
                    except Exception as e:
                        print("An error occurred while downloading pose_landmarker_lite.task:")
                        print(e)
                        return -1
            case 'full':
                if os.path.isfile(task_file_dir + 'pose_landmarker_full.task'):
                    return task_file_dir + 'pose_landmarker_full.task'
                else:
                    print("Mediapipe Pose Landmark file 'full' not yet downloaded. Downloading now into './landmark_files' sub directory.")
                    try:
                        url = (
                            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
                        )
                        filename = task_file_dir + "pose_landmarker_full.task"
                        urlretrieve(url, filename)
                        print("Downloading Mediapipe Pose Landmark file 'pose_landmarker_full.task' successful")
                        return task_file_dir + 'pose_landmarker_full.task'
                    except Exception as e:
                        print("An error occurred while downloading pose_landmarker_full.task:")
                        print(e)
                        return -1
            case 'heavy':
                if os.path.isfile(task_file_dir + 'pose_landmarker_heavy.task'):
                    return task_file_dir + 'pose_landmarker_heavy.task'
                else:
                    print("Mediapipe Pose Landmark file 'heavy' not yet downloaded. Downloading now into './landmark_files' sub directory.")
                    try:
                        url = (
                            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
                        )
                        filename = task_file_dir + "pose_landmarker_heavy.task"
                        urlretrieve(url, filename)
                        print("Downloading Mediapipe Pose Landmark file 'pose_landmarker_heavy.task' successful")
                        return task_file_dir + 'pose_landmarker_heavy.task'
                    except Exception as e:
                        print("An error occurred while downloading pose_landmarker_heavy.task:")
                        print(e)
                        return -1
            case _:
                print("The given 'landmark_type' parameter is invalid. Can be 'lite', 'full' and 'heavy'") 
                print("More information on https://developers.google.com/mediapipe/solutions/vision/pose_landmarker")
        return -1

    def create_landmark_options(self):
        """
        Define the options needed by Mediapipe Pose Landmarker.
            Params: None
            Returns: options (mediapipe.tasks.python.vision.PoseLandmarkerOptions): The defined options
        """
        landmark_path = self.check_pose_landmark_file()
        base_options = python.BaseOptions(model_asset_path=landmark_path, delegate="GPU")
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            output_segmentation_masks=False,
        )
        return options
    
    def analyze_video(self):
        """
        Analyzes a video file or webcam stream and detects human poses frame by frame using MediaPipe Pose. 

        Paramss:
            path (String): The path to the video file or '0' for webcam stream.
        Returns:
            None
        """
        frame = 0

        # Skip if already analyzed
        splitted_path = os.path.split(self.video_path)
        chars_to_cut_off = len(splitted_path[len(splitted_path)-1])
        completed_path = self.video_path[:-chars_to_cut_off]
        completed_file_name = completed_path + "pose_extraction_of_" + f"{splitted_path[len(splitted_path)-1]}".rstrip(".mp4") + "_completed"
        if os.path.isfile(completed_file_name):
            return
        
        print(f"Starting analysis of video {self.video_path}")
        with vision.PoseLandmarker.create_from_options(self.landmark_options) as landmarker:        
            cap = cv2.VideoCapture(self.video_path)
            while cap.isOpened():
                success, image = cap.read()
                start_time = time.time()
                if not success:
                    
                    # create a "completed" file for pausing analysis of CANDOR dataset when movie file is over.    
                    if frame > 0:
                        self.write_pose_to_csv()
                        file = open(completed_file_name, 'w', newline='')
                        file.close()
                    break

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                self.timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
                result = landmarker.detect_for_video(mp_image, self.timestamp_ms)
                        
                self.output_window = cv2.cvtColor(
                    self.draw_landmarks(frame, image, result), cv2.COLOR_BGR2RGB)
                
                if self.output_window is not None:
                    cv2.imshow("MediaPipe Pose Landmark", self.output_window)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return -1

                self.write_pose_to_csv()
                end_time = time.time()
                print(f"FPS of video: {1.0 / (end_time-start_time)}")
                frame += 1

    def write_pose_to_csv(self):
        """
        Writes the pose data of the person with the id person_id to the
        corresponding csv file. If no file exists, a new one is created. After
        storing the data, the global pose variable gets overwritten with "" to
        save memory and to speed up computation

        Parameters:
            Nothing
        Returns:
            Nothing
        """

        # get the csv file name
        splitted_path = self.video_path.split("/")
        chars_to_cut_off = len(splitted_path[len(splitted_path)-1])
        new_video_path = self.video_path[:-chars_to_cut_off]
        file_name = new_video_path + splitted_path[len(splitted_path)-1]+ "_posedata.csv"
        # actually write to the csv file
        if os.path.isfile(file_name):
            file = open(file_name, 'a', newline='')
            if self.analyzed_results_person.startswith("frame,person,x,y,z"):
                self.analyzed_results_person = self.analyzed_results_person.lstrip("frame,person,x,y,z\n")
            file.writelines(self.analyzed_results_person)
            self.analyzed_results_person = ""
        else:
            file = open(file_name, 'w', newline='')
            file.writelines(self.analyzed_results_person)
            self.analyzed_results_person = ""
        file.close()
        
    def draw_landmarks(self, frame, image, results):
        """
        Draws landmarks on the given image using the results from pose estimation.

        Args:
            frame (int): The current video frame 
            image (numpy.ndarray): The input image.
            results (mediapipe.python.solution_base.SolutionOutputs): The pose estimation results.

        Returns:
            numpy.ndarray: The image with landmarks drawn.
        """

        pose_landmarks_list = results.pose_landmarks
        annotated_image = np.copy(image)
        
        # draw poses on opencv window.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]
                
            # this is used to save found points for csv files
            for landmark in pose_landmarks:
                new_row = f"{frame},{idx},{landmark.x},{landmark.y},{landmark.z}"
                self.analyzed_results_person += "\n" + new_row    
                    
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()

            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z) for landmark in pose_landmarks
            ])
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                mp.solutions.pose.POSE_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_pose_landmarks_style())

        return annotated_image
