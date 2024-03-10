import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import time
import os
import sys
from urllib.request import urlretrieve

from google.cloud import storage

class PoseAnalyzer():

    def __init__(self, video_path, landmark_type):
        self.output_window = None    # Used as screen variable to draw landmarks on
        self.last_timestamp_ms = 0   # needed by Mediapipe when in "LIVE_STREAM"-RunningMode
        
        self.analyzed_results = "frame,person,x,y,z\n"  # used to store pose data of the person with the id 0 in a conversation
        self.landmark_type = landmark_type
        self.landmark_options = self.create_landmark_options()
        self.video_path = video_path
        
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
                    print("Mediapipe Pose Landmark file 'lite' already downloaded")
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
                    print("Mediapipe Pose Landmark file 'full' already downloaded")
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
                    print("Mediapipe Pose Landmark file 'heavy' already downloaded")
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
        landmark_path = self.check_pose_landmark_file()
        # define Mediapipe pose landmark detector's options
        base_options = python.BaseOptions(model_asset_path=landmark_path, delegate="GPU")
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.7,
            min_pose_presence_confidence=0.7,
            min_tracking_confidence=0.8,
            output_segmentation_masks=False,
        )
        return options
    
    def analyze_video(self):
        """
        Analyzes a video file or webcam stream and detects human poses using MediaPipe Pose. 
        First step is to get the id of that video in it's directory, i.e. if it is the video of
        the first person or the second ( id = 0 or 1).

        Args:
            path (str): The path to the video file or '0' for webcam stream.
        Returns:
            None
        """
        frame = 0
        
        with vision.PoseLandmarker.create_from_options(self.landmark_options) as landmarker:        
            cap = cv2.VideoCapture(self.video_path)
            while cap.isOpened():
                success, image = cap.read()
                start_time = time.time()
                if not success:
                    
                    # create a "completed" file for pausing analysis of CANDOR dataset when movie file is over.    
                    if frame > 0:
                        splitted_path = self.video_path.split("/")
                        chars_to_cut_off = len(splitted_path[len(splitted_path)-1])
                        completed_path = self.video_path[:-chars_to_cut_off]
                        file_name = completed_path + "pose_extraction_of_" + "{splitted_path[len(splitted_path)-1]}".rstrip(".mp4") + "_completed"
                        file = open(file_name, 'w', newline='')
                        file.close()
                    break
                if not self.detect_both_online(image):
                    frame += 1
                    continue

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                self.timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
                result = landmarker.detect_for_video(mp_image, self.timestamp_ms)
                        
                self.output_window = cv2.cvtColor(
                    self.draw_landmarks(frame, image, result), cv2.COLOR_RGB2BGR)
                
                if self.output_window is not None:
                    cv2.imshow("MediaPipe Pose Landmark", self.output_window)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return -1

                # write found pose data every x frames to corresponding csv file.
                # should be done more often since fps rate decreases      
                if frame%200 == 0:
                    self.write_pose_to_csv(file_name)
    
                end_time = time.time()
                frame += 1

    def write_pose_to_csv(self, file_name):
        """
        Writes the pose data of the person with the id person_id to the
        corresponding csv file. If no file exists, a new one is created. After
        storing the data, the global pose variable gets overwritten with "" to
        save memory and to speed up computation

        Parameters:
        file_path (String): the path of the video file whose pose data gets written
        person_id (Integer): The id of the person in the conversation, whose data gets written. Can be 0 or 1

        Returns:
        Nothing
        """
        file_name += "csv"
        global bucket
        csv_blob = bucket.blob(file_name)
        print(csv_blob)
        # get the csv file name
        #splitted_path = self.video_path.split("/")
        #chars_to_cut_off = len(splitted_path[len(splitted_path)-1])
        #new_video_path = self.video_path[:-chars_to_cut_off]
        #file_name = new_video_path + splitted_path[len(splitted_path)-3]+ f"_posedata_{id}.csv"
        
        #  write to local csv file
        if os.path.isfile(file_name):
            file = open(file_name, 'a', newline='')
            if self.analyzed_results.startswith("frame,person,x,y,z"):
                self.analyzed_results = self.analyzed_results.lstrip("frame,person,x,y,z\n")
            file.writelines(self.analyzed_results)
            self.analyzed_results = ""
            
        else:
            file = open(file_name, 'w', newline='')
            if id == 0:
                file.writelines(self.analyzed_results)
                self.analyzed_results = ""
        file.close()

        # read full new csv file to copy it
        csv_data = ""
        file = open(file_name, 'r')
        csv_data = file.read()
        file.close()


        # actually write to cloud storage
        with csv_blob.open("w") as blob_file:
            blob_file.write(csv_data)
        
    def detect_both_online(self, image):
        """
        Detects whether both screens are online based on the given image. Only need if there are 2 screens in the video.

        Parameters:
        image (numpy.ndarray): The input image.

        Returns:
        bool: True if both screens are online, False otherwise.
        """

        left_online = False
        right_online = False
        width = image.shape[1]
        left = image[:, :width//2]
        right = image[:, width//2:]
        mean_left = np.mean(left)
        #print(mean_left)
        if mean_left > 10:
            left_online = True
        if np.mean(right) > 10:
            right_online = True
        
        if left_online and right_online:
            return True
        else:
            return False

    def draw_landmarks(self, frame, id, image, results):
        """
        Draws landmarks on the given image using the results from pose estimation.

        Args:
            image (numpy.ndarray): The input image.
            person_id (int): The ID of the person of the conversation whose video is processed. Can be 0 or 1 
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
                new_row = f"{frame},{id},{landmark.x},{landmark.y},{landmark.z}"
                if id == 0:
                    self.analyzed_results_person_0 += "\n" + new_row
                else:
                    self.analyzed_results_person_1 += "\n" + new_row    
                    
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

def grab_videos_from_storage(video_path):
    global bucket
    video_blob = bucket.blob(video_path)
    video_blob.download_to_filename(local_file_name)


if __name__ == "__main__":
    video_path = "Path"
    storage_adress = sys.argv[1]
    local_file_name = sys.argv[2]
    landmark_type = 'lite'
    storage_client = storage.Client()
    bucket = storage_client.bucket(storage_adress)
    grab_videos_from_storage(video_path)    
    pose_analyzer = PoseAnalyzer(local_file_name, landmark_type)
    pose_analyzer.analyze_video()