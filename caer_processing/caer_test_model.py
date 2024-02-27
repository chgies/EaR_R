import cv2
import numpy as np
import torch
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from models.CAER_model_training import EmotionV0
from caer_feature_extractor import CAERFeatureExtractor
import os
current_points = []
detector = 0
emotion_model = 0

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def load_model_weights(model, filepath='./caer_processing/models/CAER_model_weights.pth'):
    """
    Load the saved CAER model weights
        Params:
            model (EmotionV0): The EmotionV0 model object that the weights should be loaded to
            filepath (String): The path to the saved model weights
    """
    model.load_state_dict(torch.load(filepath, map_location=torch.device(device)))
    print(f'Model weights loaded from {filepath}')


def draw_landmarks(frame_index, input_image, results):
    """
    Draws landmarks on the given image using the results from pose estimation.
        Params:
            frame_index (int): The current video frame
            input_image (numpy.ndarray): The input image. 
            results (mediapipe.python.solution_base.SolutionOutputs): The pose estimation results.
        Returns:
            annotated_image (numpy.ndarray): The image with landmarks drawn.
    """
    global current_points
    pose_landmarks_list = results.pose_landmarks
    annotated_image = np.copy(input_image)
    if len(pose_landmarks_list) > 0:
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]
            for landmark in pose_landmarks:
                new_row = [frame_index,idx,landmark.x,landmark.y,landmark.z]
                current_points.append(new_row)    
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

def prepare_loop():
    global emotion_model, detector, device
    # Create model instance
    emotion_model = EmotionV0(57,104,7)
    load_model_weights(emotion_model)

    #Init Mediapipe Landmarker
    base_options = python.BaseOptions(model_asset_path='./landmark_files/pose_landmarker_lite.task', delegate="GPU")
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

def run_video_loop():
    global detector, current_points
    FRAME_BUFFER_MAX_SIZE = 45
    frame_buffer = []
    frame_index = 0
    cap = cv2.VideoCapture(0)
    while True:
        success, image = cap.read()
        if not success:
            break
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, 
            data=image)
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        detection_result = detector.detect_for_video(mp_image, timestamp_ms)
        if detection_result is not None:
            output_window = cv2.cvtColor(
                    draw_landmarks(frame_index, image, detection_result), cv2.COLOR_BGR2RGB)
            frame_index += 1
        else:
            cv2.imshow("MediaPipe Pose Landmark", image)
            continue
        if frame_index == FRAME_BUFFER_MAX_SIZE:
            if len(current_points) > 0:
                frame_buffer = np.asarray(current_points)
                frame_buffer_as_dataframe = pd.DataFrame(data=frame_buffer, columns=['frame','person','x','y','z'])
                current_feature_extractor = CAERFeatureExtractor(frame_buffer_as_dataframe, False)
                calculated_values = current_feature_extractor.get_element_list_as_dataframes()
                frame_index = -5
                calc_values = np.asarray(calculated_values.iloc[1:,].values, dtype=np.float32)
                emotion_as_tensor = torch.tensor(calc_values, dtype=torch.float32).to(device)
                emotion = emotion_model(emotion_as_tensor)
                print(emotion)
                current_points = []
        if output_window is not None:
            cv2.imshow("MediaPipe Pose Landmark", cv2.cvtColor(output_window, cv2.COLOR_RGB2BGR))
        frame_index += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

prepare_loop()
run_video_loop()