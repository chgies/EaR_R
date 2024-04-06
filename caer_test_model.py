import cv2
import numpy as np
import torch
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from caer_processing.models.emotionV0.EmotionV0 import EmotionV0
from caer_processing.models.emotionV1.EmotionV1 import EmotionV1
from caer_processing.models.emotionV2.EmotionV2 import EmotionV2
from caer_processing.models.emotionV3.EmotionV3 import EmotionV3
from caer_processing.caer_feature_extractor import CAERFeatureExtractor

MODEL_TO_TEST = "EmotionV3"

current_points = []
detector = 0
emotion_model = 0

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def load_model_weights(model):
    """
    Load the saved CAER model weights
        Params:
            model (EmotionV0): The EmotionV0 model object that the weights should be loaded to
            filepath (String): The path to the saved model weights
    """
    match MODEL_TO_TEST:
        case "EmotionV0":
            filepath='./caer_processing/models/emotionV0/CAER_model_weights.pth'
        case "EmotionV1":
            filepath='./caer_processing/models/emotionV1/CAER_model_weights.pth'
        case "EmotionV2":
            filepath='./caer_processing/models/emotionV2/CAER_model_weights.pth'
        case "EmotionV3":
            filepath='./caer_processing/models/emotionV3/CAER_model_weights.pth'    
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
    """
    Initialize the emotion model and the Mediapipe Task Landmarker needed to run the test lpp function.
        Params:
            None
        Returns:
            None 
    """
    global MODEL_TO_TEST, emotion_model, detector, device
    # Create model instance
    match MODEL_TO_TEST:
        case "EmotionV0":
            emotion_model = EmotionV0(51,104,7)
        case "EmotionV1":
            emotion_model = EmotionV1(51,104,7)
        case "EmotionV2":
            emotion_model = EmotionV2(37,60,7)
        case "EmotionV3":
            emotion_model = EmotionV3(22,35,7)
    load_model_weights(emotion_model)

    #Init Mediapipe Landmarker
    base_options = python.BaseOptions(model_asset_path='./landmark_files/pose_landmarker_heavy.task', delegate="GPU")
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        output_segmentation_masks=False,
    )
    detector = vision.PoseLandmarker.create_from_options(options)


def run_video_loop():
    """
    Run a loop of testing the trained emotion model with your own webcam unsing Mediapipe Pose Landmarker.
        Params:
            None
        Returns:
            None
    """
    global detector, current_points, emotion_model
    FRAME_BUFFER_MAX_SIZE = 45
    frame_buffer_full = False
    frame_buffer = []
    frame_index = 0
    cap = cv2.VideoCapture(0)
    emotion_as_word = ""
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
        else:
            cv2.imshow("MediaPipe Pose Landmark", image)
            continue
        print(frame_index)
        if frame_index == FRAME_BUFFER_MAX_SIZE:
            if len(current_points) > 0:
                frame_buffer = np.asarray(current_points)
                frame_buffer_as_dataframe = pd.DataFrame(data=frame_buffer, columns=['frame','person','x','y','z'])
                current_feature_extractor = CAERFeatureExtractor(frame_buffer_as_dataframe, False)
                calculated_values = current_feature_extractor.get_element_list_as_dataframes()
                frame_index = -5
                calc_values = np.asarray(calculated_values.iloc[1:,].values, dtype=np.float32)
                match MODEL_TO_TEST:
                    case "EmotionV2":
                        calc_values = np.delete(calc_values, [14,15,16,18,19,21,22,23,25,26,27,46,47,46,50],axis=1)
                    case "EmotionV3":
                        calc_values = np.delete(calc_values, [3,7,11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,27,30,31,34,38,39,42,43,46,47,49,50],axis=1)
                calc_values_as_tensor = torch.tensor(calc_values, dtype=torch.float32).to(device)
                emotion_as_tensor = emotion_model(calc_values_as_tensor)
                print(f"max value in tensor: {torch.argmax(emotion_as_tensor)}")
                emotion_as_value = torch.argmax(emotion_as_tensor).item()
                match emotion_as_value:
                    case 1: emotion_as_word = "Anger"
                    case 2: emotion_as_word = "Disgust"
                    case 3: emotion_as_word = "Fear"
                    case 4: emotion_as_word = "Happy"
                    case 5: emotion_as_word = "Sad"
                    case 6: emotion_as_word = "Surprise"
                    case 7: emotion_as_word = "Neutral"
                    case _: emotion_as_word = str(emotion_as_value)
                print(f"Emotion: {emotion_as_value}")
                current_points = []
                frame_buffer_full = True
                frame_index = 0
        if output_window is not None:
            cv2.putText(output_window, emotion_as_word, (int(output_window.shape[0]/2), 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow("MediaPipe Pose Landmark", cv2.cvtColor(output_window, cv2.COLOR_RGB2BGR))
        if not frame_buffer_full:
            frame_index += 1
        else:
            frame_buffer_full = False
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


prepare_loop()
run_video_loop()