import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import pandas as pd
import time
import os
from urllib.request import urlretrieve

class CANDORPoseAnalyzer():

    def __init__(self, landmark_type, dir_to_extract, show_fps, show_video):
        """
        The CANDORPoseAnalyzer class holds the methods to extract the pose coordinates out of the CANDOR dataset. It utilizes the
        MediaPipe Pose Landmarker solution to extract the joint points of the body in every frames. When calling the drawing function,
        the coordinates get saved as strings into a variable in every frame. This data is then stored in a corresponding csv file in
        the directory given by the user.
        
        Parameters:
            None
        Returns:
            The path to the file in string format
        """
        self.dir_to_extract = dir_to_extract
        self.landmark_type = landmark_type
        self.show_fps = show_fps
        self.show_video = show_video
        self.landmark_options = self.create_landmark_options()

        
    def load_feature_file(self, video_path, person_id):
        """
        Read the corresponding "audio_video_features.csv" file of the movie and filter out its biggest emotion label. 
        Since the csv shows one line per second of the movie, this function expands this to 30 frames per second,
        resulting in one csv line being converted to 30 feature lines.

            Params:
                video_path (String): The path to the video file
                person_id (String): The id of the person in the video
            Returns:
                feature_list(List): A list of emotional labels, 1 element stands for 1 frame (1/30 sec) of the movie
        """
        feature_file = os.path.split(video_path)[0]
        feature_file = f"{feature_file[:-10]}/audio_video_features.csv"
        feature_data = pd.read_csv(feature_file)
        feature_list = []
        frame = 0
        for row in feature_data.iloc():
            if row[1] == person_id:    
                for frame_of_second in range(frame,frame+30):
                        emotion_index = np.argmax([row[45], row[46], row[47], row[48],row[49],row[50],row[51],row[52]])
                        emotion_row = [frame_of_second, emotion_index]
                        feature_list.append(emotion_row)
                frame = frame_of_second+1        
        return feature_list

    def check_pose_landmark_file(self):
        """
        This function checks if the needed Mediapipe Pose estimation landmark files are available in the sub directory
        '/landmark_files'. If not, they get downloaded from Mediapipe servers. 
        More info on https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

        Parameters:
            None
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
            Params:
                None
            Returns: 
                options (mediapipe.tasks.python.vision.PoseLandmarkerOptions): The defined options used to create the Landmarker instance
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
    
    def analyze_video(self, video_path):
        """
        Analyzes a video file or webcam stream and detects human poses frame by frame using MediaPipe Pose. 

        Params:
            video_path (String): The path to the video file.
        Returns:
            None
        """
        frame = 0

        output_window = None    # Used as screen variable to draw landmarks on
        last_timestamp_ms = 0   # needed by Mediapipe when in "LIVE_STREAM"-RunningMode        
        analyzed_results_person = "frame,person,x,y,z,visibility,emotion\n"  # used to store pose data of the person with the id 0 in a conversation        
        person_id = os.path.split(video_path)[1][:-4]
        feature_list = self.load_feature_file(video_path, person_id)
        
        # if new video has begun and a pose csv file already exists, this means that an extraction run before was aborted and 
        # the video wasn't fully analyzed (If it was, then it wouln't be on the list, bc it would be saved in the "extracted_videos.txt" 
        # file in the user-defined pose extraction directory). Writing new values into the old file causes errors, thus delete the old file.
        csv_file_name = self.dir_to_extract + "/" + person_id + "_posedata.csv"
        if os.path.isfile(csv_file_name):
            os.remove(csv_file_name)

        print(f"Starting analysis of video {video_path}")
        with vision.PoseLandmarker.create_from_options(self.landmark_options) as landmarker:        
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                success, image = cap.read()
                if self.show_fps:
                    start_time = time.time()
                if not success:
                    if frame > 0:
                        analyzed_results_person = self.write_pose_to_csv(person_id, analyzed_results_person)
                        already_extracted_file = f"{self.dir_to_extract}/extracted_videos.txt"
                        if not os.path.isfile(already_extracted_file):
                            file = open(already_extracted_file, 'w')
                            file.write(f"{person_id}.mp4")
                            file.close
                        else:
                            file = open(already_extracted_file, 'a')
                            file.write(f"\n{person_id}.mp4")
                            file.close
                    break   
                # only search for poses when an emotion probability is defined in 'audio_video_features.csv' file for this frame
                if not feature_list[frame] == "":
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=image)
                    self.timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
                    result = landmarker.detect_for_video(mp_image, self.timestamp_ms)
                    analyzed_results_person, output_window = self.draw_landmarks(frame, image, feature_list, analyzed_results_person, result)
                    if self.show_video and output_window is not None:
                        cv2.imshow("MediaPipe Pose Landmark", output_window)
                else:                
                    if self.show_video:
                        cv2.imshow("MediaPipe Pose Landmark", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return -1
                if frame%200 == 0:
                    analyzed_results_person = self.write_pose_to_csv(person_id, analyzed_results_person)
                if self.show_fps:
                    end_time = time.time()
                    print(f"FPS of video {person_id}.mp4: {1.0 / (end_time-start_time)}")
                frame += 1

    def write_pose_to_csv(self, person_id, csv_results):
        """
        Writes the pose data of the person with the id person_id to the
        corresponding csv file. If no file exists, a new one is created. After
        storing the data, the global pose variable gets overwritten with "" to
        save memory and to speed up computation

        Params:
            person_id (String): The id of the person in the video
            csv_results (String): The pose coordinates for every frame that are written to the csv file
        Returns:
            csv_results (String): The variable that holds the pose coordinates, now empty and ready to fill with coordinates again
        """
        file_name = self.dir_to_extract + "/" + person_id + "_posedata.csv"
        # actually write to the csv file
        if os.path.isfile(file_name):
            file = open(file_name, 'a', newline='')
            if csv_results.startswith("frame,person,x,y,z"):
                csv_results = csv_results.lstrip("frame,person,x,y,z,visibility,emotion\n")
            file.writelines(csv_results)
            csv_results = ""
        else:
            file = open(file_name, 'w', newline='')
            file.writelines(csv_results)
            csv_results = ""
        file.close()
        return csv_results
        
    def draw_landmarks(self, frame, image, feature_list, csv_results, results):
        """
        Draws landmarks on the given image using the results from pose estimation.

        Args:
            frame (int): The current video frame number
            image (numpy.ndarray): The input image as numpy array.
            feature_list (List): The list containing the emotion probabilities for the person for every frame
            csv_results (String): The current list of pose estimation coordinates and emotions that is later written to the csv file
            results (mediapipe.python.solution_base.SolutionOutputs): The MediaPipe pose estimation results for the current frame.

        Returns:
            csv_results (String): The current list of pose coordinates, with added coordinates and emotion probabilities for this frame
            annotated_image (numpy.ndarray): The image with landmarks drawn as numpy array.
        """
        pose_landmarks_list = results.pose_landmarks
        annotated_image = np.copy(image)
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]
            # this is used to save found points for csv files
            for landmark in pose_landmarks:
                new_row = f"{frame},{idx},{landmark.x},{landmark.y},{landmark.z},{round(landmark.visibility,2)}, {feature_list[frame][1]}"
                csv_results += "\n" + new_row    
            # draw poses on opencv window.
            if self.show_video:
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
        return csv_results, annotated_image
