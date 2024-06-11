import cv2
import numpy as np
import torch
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
#from caer_processing.models.emotionV0.EmotionV0 import EmotionV0
#from caer_processing.models.emotionV1.EmotionV1 import EmotionV1
#from caer_processing.models.emotionV50.EmotionV50 import EmotionV50
#from caer_processing.models.emotionV80.EmotionV80 import EmotionV80
from candor_processing.candor_feature_extractor import CANDORFeatureExtractor

# Choose the model you like to test. Possible models are "EmotionV0", "EmotionV1", "EmotionV50" and "EmotionV80"
MODEL_TO_TEST = "EmotionV0"

# Choose if you want to train the net with features following Aristidou (2015, aee references folder), or high level Laban motor elements
USE_LABAN_FEATURES = False

# Global variables
current_points = []
detector = 0
emotion_model = 0
features_to_delete = []

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def load_model_configuration(model_type):
    """
    Load the saved CAER model weight filepath and the list of features that need to be ignored
        Params:
            model (EmotionV0): The EmotionV0 model object that the weights should be loaded to
            filepath (String): The path to the saved model weights
    """
    match model_type:
        case "EmotionV0":
            if not USE_LABAN_FEATURES:
                filepath='./caer_processing/models/emotionV0/CAER_model_weights.pth'
            else:
                filepath='./caer_processing/models/emotionV0/CAER_laban_model_weights.pth'
            feature_list = []    
        case "EmotionV1":
            if not USE_LABAN_FEATURES:
                filepath='./caer_processing/models/emotionV1/CAER_model_weights.pth'
            else:
                filepath='./caer_processing/models/emotionV1/CAER_laban_model_weights.pth'
            read_file = open("./caer_processing/models/emotionV1/feature.list")
            line = read_file.read().replace("[", "").replace("]","").replace(",","")
            feature_list = [int(element) for element in line.split()]
        case "EmotionV50":
            if not USE_LABAN_FEATURES:
                filepath='./caer_processing/models/emotionV50/CAER_model_weights.pth'
            else:
                filepath='./caer_processing/models/emotionV50/CAER_laban_model_weights.pth'
            read_file = open("./caer_processing/models/emotionV50/feature.list")
            line = read_file.read().replace("[", "").replace("]","")
            feature_list = [int(element) for element in line.split()]    
        case "EmotionV80":
            if not USE_LABAN_FEATURES:
                filepath='./caer_processing/models/emotionV80/CAER_model_weights.pth'
            else:
                filepath='./caer_processing/models/emotionV80/CAER_laban_model_weights.pth'
            read_file = open("./caer_processing/models/emotionV80/feature.list")
            line = read_file.read().replace("[", "").replace("]","")
            feature_list = [int(element) for element in line.split()]        
    return filepath, feature_list

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
    global MODEL_TO_TEST, emotion_model, detector, device, features_to_delete
    # Create model instance
    '''weight_filepath, features_to_delete = load_model_configuration(MODEL_TO_TEST)
    if not USE_LABAN_FEATURES:
        num_of_features = 51 - len(features_to_delete)
    else:
        num_of_features = 26 - len(features_to_delete)
    match MODEL_TO_TEST:
        case "EmotionV0":
            if not USE_LABAN_FEATURES:
                emotion_model = EmotionV0(num_of_features,num_of_features*2,7)
            else:
                emotion_model = EmotionV0(num_of_features,num_of_features*2,7)
        case "EmotionV1":
            emotion_model = EmotionV1(num_of_features,num_of_features*2,7)
        case "EmotionV50":
            emotion_model = EmotionV50(num_of_features,num_of_features*2,7)
        case "EmotionV80":
            emotion_model = EmotionV80(num_of_features,num_of_features*2,7)
    emotion_model.load_state_dict(torch.load(weight_filepath, map_location=torch.device(device)))
    '''
    #Init Mediapipe Landmarker

    # Create a pose landmarker instance with the live stream mode:
    def print_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            # cv2.imshow('Show', output_image.numpy_view())
            print('pose landmarker result: {}'.format(result))

    base_options = python.BaseOptions(model_asset_path='./landmark_files/pose_landmarker_heavy.task', delegate="GPU")
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        output_segmentation_masks=False,
        result_callback = print_result
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
    global detector, current_points, emotion_model, features_to_delete
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
        detection_result = detector.detect_async(mp_image, timestamp_ms)
        if detection_result is not None:
            output_window = cv2.cvtColor(
                    draw_landmarks(frame_index, mp_image, detection_result), cv2.COLOR_BGR2RGB)
        else:
            cv2.imshow("MediaPipe Pose Landmark", image)
            continue
        #print(frame_index)
        if frame_index == FRAME_BUFFER_MAX_SIZE:
            if len(current_points) > 0:
                frame_buffer = np.asarray(current_points)
                frame_buffer_as_dataframe = pd.DataFrame(data=frame_buffer, columns=['frame','person','x','y','z'])
                current_feature_extractor = CANDORFeatureExtractor(frame_buffer_as_dataframe, USE_LABAN_FEATURES, False)
                if not USE_LABAN_FEATURES:
                    calculated_values = current_feature_extractor.get_element_list_as_dataframes()
                else:
                    calculated_values = current_feature_extractor.get_laban_movement_list_as_dataframes()
                frame_index = -5
                calc_values = np.asarray(calculated_values.iloc[1:,].values, dtype=np.float32)
                calc_values = np.delete(calc_values, features_to_delete,axis=1)
                #calc_values_as_tensor = torch.tensor(calc_values, dtype=torch.float32).to(device)
                '''
                emotion_as_tensor = emotion_model(calc_values_as_tensor)
                #print(f"tensor: {emotion_as_tensor[0]}")
                #print(f"max value in tensor: {torch.argmax(emotion_as_tensor[0])}")
                emotion_as_value = torch.argmax(emotion_as_tensor[0]).item()
                if emotion_as_value == 0:
                    emotion_as_word = "Anger"
                elif emotion_as_value == 1:
                    emotion_as_word = "Disgust"
                elif emotion_as_value == 2:
                    emotion_as_word = "Fear"
                elif emotion_as_value == 3:
                    emotion_as_word = "Happy"
                elif emotion_as_value == 4:
                    emotion_as_word = "Sad"
                elif emotion_as_value == 5: 
                    emotion_as_word = "Surprise"
                elif emotion_as_value == 6:
                    emotion_as_word = "Neutral"
                else: 
                    emotion_as_word = str(emotion_as_value)
                #print(f"Emotion: {emotion_as_value}")
                '''
                current_points = []
                frame_buffer_full = True
                frame_index = 0
        if output_window is not None:
            #cv2.putText(output_window, emotion_as_word, (int(output_window.shape[0]/2), 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow("MediaPipe Pose Landmark", cv2.cvtColor(output_window, cv2.COLOR_RGB2BGR))
        if not frame_buffer_full:
            frame_index += 1
        else:
            frame_buffer_full = False
        if frame_index == 100:
            frame_index = 0
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


prepare_loop()
run_video_loop()