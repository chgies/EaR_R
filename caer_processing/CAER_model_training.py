import pandas as pd
import numpy as np
import torch
import os
from sklearn.ensemble import RandomForestClassifier
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from models.emotionV0.EmotionV0 import EmotionV0
from models.emotionV1.EmotionV1 import EmotionV1
from models.emotionV50.EmotionV50 import EmotionV50
from models.emotionV80.EmotionV80 import EmotionV80

# Choose, which model type to train. Possible: "EmotionV0", "EmotionV1", "EmotionV50", "EmotionV80" 
MODEL_TO_TRAIN = "EmotionV0"

# Choose if you want to train the net with features following Aristidou (2015, aee references folder), or high level Laban motor elements
USE_LABAN_FEATURES = False

# Choose, if you want to use Random Forest Classification to sort features by importance and delete unimportant features.
# can be used with EmotionV1, EmotionV50 and EmotionV80 
AUTO_SORT_BY_IMPORTANCE = True

# Choose, if you want to normalize all dataset values and train the net with different values.
# Doesn't work while USE_LABAN_FEATURES is True
CREATE_NORMALIZED_CSV = False

CAER_DIR = os.environ["CAER_DIR"]
NUM_FEATURES = 0
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def train_and_test_model():
    """
    Run the loop of preparing the training data, train the model and evaluate it.
        Params:
            None
        Returns:
            None
    """
    global NUM_FEATURES
    if USE_LABAN_FEATURES == False:
        use_laban_features = False
        add_to_path_text = ""
    else:
        use_laban_features = True
        add_to_path_text = "_laban"
    if CREATE_NORMALIZED_CSV:
        operations = ['extracted', 'normalized', 'rescaled', 'standardized']
        normalize_extracted_values(f"{CAER_DIR}/train/extracted_train_values.csv", f"{CAER_DIR}/test/extracted_test_values.csv")
    else:
        operations = ['extracted']
    for operation in operations:

        train_file_path = f"{CAER_DIR}/train/{operation}{add_to_path_text}_train_values.csv"  
        test_file_path = f"{CAER_DIR}/test/{operation}{add_to_path_text}_test_values.csv"  
        df_train = pd.read_csv(train_file_path)
        X_train_np_array = np.asarray(df_train.iloc[1:, 1:-1].values, dtype=np.float32)    
        y_train_np_array = np.asarray(df_train.iloc[1:, -1:].values, dtype=np.float32)
        df_test = pd.read_csv(test_file_path)
        X_test_np_array = np.asarray(df_test.iloc[1:, 1:-1].values, dtype=np.float32)
        if use_laban_features == False:
            features_to_delete_for_v1 = [0,1,3,4,5,7,8,9,11,12,13,16,17,19,20,22,24,26,28,29,31,32,33,35,36,37,39,40,41,43,44,45,47,50]   
        else:
            features_to_delete_for_v1 = []
        if AUTO_SORT_BY_IMPORTANCE:
            features_to_delete_for_50, features_to_delete_for_80 = calculate_least_important_features(df_test, operation)
        else:
            features_to_delete_for_50, features_to_delete_for_80 = ([],[])
        # remove unused features in training dataset
        match MODEL_TO_TRAIN:
            case "EmotionV1":
                X_train_np_array = np.delete(X_train_np_array, features_to_delete_for_v1,axis=1)
            case "EmotionV80":
                match operation:
                    case 'extracted':
                        X_train_np_array = np.delete(X_train_np_array, features_to_delete_for_80,axis=1)
                    case 'normalized':
                        X_train_np_array = np.delete(X_train_np_array, features_to_delete_for_80,axis=1)
                    case 'rescaled':
                        X_train_np_array = np.delete(X_train_np_array, features_to_delete_for_80,axis=1)
                    case 'standardized':
                        X_train_np_array = np.delete(X_train_np_array, features_to_delete_for_80,axis=1)
            case "EmotionV50":
                match operation:
                    case 'extracted':
                        X_train_np_array = np.delete(X_train_np_array, features_to_delete_for_50,axis=1)
                    case 'normalized':
                        X_train_np_array = np.delete(X_train_np_array, features_to_delete_for_50,axis=1)
                    case 'rescaled':
                        X_train_np_array = np.delete(X_train_np_array, features_to_delete_for_50,axis=1)
                    case 'standardized':
                        X_train_np_array = np.delete(X_train_np_array, features_to_delete_for_50,axis=1)                         
        NUM_FEATURES = X_train_np_array.shape[1]
        X_train = torch.tensor(X_train_np_array, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train_np_array, dtype=torch.float32).to(device)
        
        # remove unused features in testing dataset
        match MODEL_TO_TRAIN:
            case "EmotionV1":
                X_test_np_array = np.delete(X_test_np_array, features_to_delete_for_v1,axis=1)
            case "EmotionV80":
                match operation:
                    case 'extracted':
                        X_test_np_array = np.delete(X_test_np_array, features_to_delete_for_80,axis=1)
                    case 'normalized':
                        X_test_np_array = np.delete(X_test_np_array, features_to_delete_for_80,axis=1)
                    case 'rescaled':
                        X_test_np_array = np.delete(X_test_np_array, features_to_delete_for_80,axis=1)
                    case 'standardized':
                        X_test_np_array = np.delete(X_test_np_array, features_to_delete_for_80,axis=1)
            case "EmotionV50":
                match operation:
                    case 'extracted':
                        X_test_np_array = np.delete(X_test_np_array, features_to_delete_for_50,axis=1)
                    case 'normalized':
                        X_test_np_array = np.delete(X_test_np_array, features_to_delete_for_50,axis=1)
                    case 'rescaled':
                        X_test_np_array = np.delete(X_test_np_array, features_to_delete_for_50,axis=1)
                    case 'standardized':
                        X_test_np_array = np.delete(X_test_np_array, features_to_delete_for_50,axis=1)                         
        y_test_np_array = np.asarray(df_test.iloc[1:, -1:].values, dtype=np.float32)
        X_test = torch.tensor(X_test_np_array, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test_np_array, dtype=torch.float32).to(device)
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_dl = DataLoader(train_dataset, batch_size =5500, shuffle=True)
        test_dl = DataLoader(test_dataset, batch_size = 5500, shuffle=True)
        num_of_epochs = 20
        # Put data to target device
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_test, y_test = X_test.to(device), y_test.to(device)
        #torch.manual_seed(42)
        ### Training
        match MODEL_TO_TRAIN:
            case "EmotionV0":
                training_model = EmotionV0(NUM_FEATURES,NUM_FEATURES*2,7).to(device)
                deleted_feature_list = []
            case "EmotionV1":
                training_model = EmotionV1(NUM_FEATURES,NUM_FEATURES*2,7).to(device)
                deleted_feature_list = features_to_delete_for_v1
            case "EmotionV80":
                training_model = EmotionV80(NUM_FEATURES,NUM_FEATURES*2,7).to(device)
                deleted_feature_list = features_to_delete_for_80
            case "EmotionV50":
                training_model = EmotionV50(NUM_FEATURES,NUM_FEATURES*2,7).to(device)
                deleted_feature_list = features_to_delete_for_50
        train_model(operation, training_model, train_dl, num_of_epochs, learning_rate=0.0001)
        save_model_weights(training_model, deleted_feature_list, use_laban_features)
        # Evaluate the model on the validation set
        match MODEL_TO_TRAIN:
            case "EmotionV0":
                val_model = EmotionV0(NUM_FEATURES,NUM_FEATURES*2,7).to(device)
            case "EmotionV1":
                val_model = EmotionV1(NUM_FEATURES,NUM_FEATURES*2,7).to(device)
            case "EmotionV80":
                val_model = EmotionV80(NUM_FEATURES,NUM_FEATURES*2,7).to(device)
            case "EmotionV50":
                val_model = EmotionV50(NUM_FEATURES,NUM_FEATURES*2,7).to(device)
        weight_path, features_to_delete = load_model_configuration(MODEL_TO_TRAIN, use_laban_features)
        val_model.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))
        evaluate_model(operation, val_model, test_dl)


# Function to convert integer labels to one-hot encoded vectors
def convert_to_one_hot(labels, num_classes):
    """
    Convert the "emotion" column into 7 columns for model training
        Params:
            labels (Integer): The emotion labels as Integer, from 1 to 7
            num_classes: The amount of classes that the Model needs to find
        Returns:
            one_hot (Tensor of float): The labels splitted into a tensor of 7 x 7 values
    """
    one_hot = torch.nn.functional.one_hot(labels - 1, num_classes).float()
    return one_hot.view(-1, num_classes)

def train_model(operation, model, data_loader, num_epochs=100, learning_rate=0.001):
    """
    Train the model with the given arguments as parameters
        Params:
            operation (String): Which version of the dataset is trained. Can be 'extracted' (the original one), 'normalized', 'standardized', 'rescaled'
            model (torch.nn.Module): The model that should be trained
            input_size (int): How much features the model has
            hidden_size (int): The size of the hidden layers
            output_size (int): The amount of labels (classes) that form the output
            data_loader (torch.utils.data.DataLoader): The Training input data formated into Tensors with a maximum chunk size
            num_epochs (int): The number of training cycles
            learning_rate (float): The learning rate used to train the model
    """
    # Initialize the loss function, and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_weights  = 0
    best_accuracy = 0
    # Training loop
    for epoch in range(num_epochs):
        for inputs, labels in data_loader:
            """

            Only for testing
            
            min = torch.min(labels)
            max = torch.max(labels)
            mean = torch.mean(labels)
            median = torch.median(labels.float()).item()
            print(f"label values: {min} to {max}, mean is {mean}, median is {median}")
            """
            # Convert integer labels to one-hot encoded vectors
            labels_one_hot = convert_to_one_hot(labels.to(torch.int64), 7)
            # Forward pass
            outputs = model.forward(inputs)
            outputs = outputs.to(device)
            loss = criterion(outputs, labels_one_hot)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accuracy = (torch.argmax(outputs, 1) == torch.argmax(labels, 1)).float().mean()
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = model.state_dict()
        # Print loss for every epoch
        if (epoch + 1) % 1 == 0:
            print(f'Training {operation} {MODEL_TO_TRAIN}, epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    print('Training complete!')


def save_model_weights(model, features_to_delete, use_laban_features):
    """
    Save the trained weights into a file
        Params:
            model (torch.nn.Module): An instance of the model that has been trained
            NUM_FEATURES: The list of features that should be removed from analyzing when using the model
        Returns:
            None
    """
    match MODEL_TO_TRAIN:
        case "EmotionV0":
            if use_laban_features == False:
                filepath='./caer_processing/models/emotionV0/CAER_model_weights.pth'
            else:
                filepath='./caer_processing/models/emotionV0/CAER_laban_model_weights.pth'
        case "EmotionV1":
            if use_laban_features == False:
                filepath='./caer_processing/models/emotionV1/CAER_model_weights.pth'
            else:
                filepath='./caer_processing/models/emotionV1/CAER_laban_model_weights.pth'
            list_file = open(f"{os.path.split(filepath)[0]}/feature.list", 'w')
            list_file.write(str(features_to_delete))
            list_file.close()
        case "EmotionV80":
            if use_laban_features == False:
                filepath='./caer_processing/models/emotionV80/CAER_model_weights.pth'
            else:
                filepath='./caer_processing/models/emotionV80/CAER_laban_model_weights.pth'
            list_file = open(f"{os.path.split(filepath)[0]}/feature.list", 'w')
            list_file.write(str(features_to_delete))
            list_file.close()
        case "EmotionV50":
            if use_laban_features == False:
                filepath='./caer_processing/models/emotionV50/CAER_model_weights.pth'
            else:
                filepath='./caer_processing/models/emotionV50/CAER_laban_model_weights.pth'
            list_file = open(f"{os.path.split(filepath)[0]}/feature.list", 'w')
            list_file.write(str(features_to_delete))
            list_file.close()
    torch.save(model.state_dict(), filepath)
    print(f'Model weights saved to {filepath}')


def load_model_configuration(model_type, use_laban_features):
    """
    Load saved weights and features that need to be deleted and returns them
        Params:
            model (torch.nn.Module): An instance of the model that needs these weights
    Returns:
        filepath: The filepath of saved weights
        feature_list (list): A list of features that need to be deleted from analyzing
    """
    match model_type:
        case "EmotionV0":
            if use_laban_features == False:
                filepath='./caer_processing/models/emotionV0/CAER_model_weights.pth'
            else:
                filepath='./caer_processing/models/emotionV0/CAER_laban_model_weights.pth'
            feature_list = []
        case "EmotionV1":
            if use_laban_features == False:    
                filepath='./caer_processing/models/emotionV1/CAER_model_weights.pth'
            else:
                filepath='./caer_processing/models/emotionV1/CAER_laban_model_weights.pth'
            read_file = open("./caer_processing/models/emotionV1/feature.list")
            line = read_file.read().replace("[", "").replace("]","").replace(",","")
            feature_list = [int(element) for element in line.split()]
        case "EmotionV80":
            if use_laban_features == False:
                filepath='./caer_processing/models/emotionV80/CAER_model_weights.pth'
            else:
                filepath='./caer_processing/models/emotionV80/CAER_laban_model_weights.pth'
            read_file = open("./caer_processing/models/emotionV80/feature.list")
            line = read_file.read().replace("[", "").replace("]","")
            feature_list = [int(element) for element in line.split()]
        case "EmotionV50":
            if use_laban_features == False:
                filepath='./caer_processing/models/emotionV50/CAER_model_weights.pth'
            else:
                filepath='./caer_processing/models/emotionV50/CAER_laban_model_weights.pth'
            read_file = open("./caer_processing/models/emotionV50/feature.list")
            line = read_file.read().replace("[", "").replace("]","")
            feature_list = [int(element) for element in line.split()]    
    return filepath, feature_list


def evaluate_model(operation, model, val_loader):
    """
    Evaluate the recently trained model
        Params:
            model (torch.nn.Module): The model that is evaluated
            val_loader(torch.utils.data.DataLoader): The validation input data, splitted into chunks and formatted into Tensor format
        Returns:
            None
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.float()
            # Convert integer labels to one-hot encoded vectors
            labels_one_hot = convert_to_one_hot(labels.to(torch.int64), num_classes=7)
            # Forward pass
            outputs = model(inputs)
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == torch.argmax(labels_one_hot, dim=1)).sum().item()
    accuracy = correct / total
    print(f'Accuracy on {operation} validation set for model type {MODEL_TO_TRAIN}: {accuracy * 100:.2f}%')


def normalize_extracted_values(train_path, test_path):
    """
    Load a csv file with extracted pose features and use normalization operations on its features.
    After that, the new dataframes get saved in the same directory as the input file
    Used Operations:
        Normalization: Values get scaled to float values between 0 and 1
        Rescaling: Values get scaled to float values between -1 and 1
        Z-Score standdardization: Calculate the mean value of the data, use this as 0 value and calculate the deviation of every value
                                    to float values between -1 and 1

        Params:
            train_path (String): The path to the extracted feature csv file of the training dataset
            test_path (String): The path to the extracted feature csv file of the test dataset
        
        Returns:
            Nothing.
    """
    train_values = pd.read_csv(train_path)
    test_values = pd.read_csv(test_path)    
    normalizations = ['normalized', 'rescaled', 'standardized']
    for operation in normalizations:
        match operation:
            case 'normalized':
                changed_train_values = train_values.copy()
                changed_train_values = changed_train_values.drop("Unnamed: 0",  axis=1)
                changed_test_values = test_values.copy()
                changed_test_values = changed_test_values.drop("Unnamed: 0",  axis=1)
                for column in changed_train_values.columns: 
                    changed_train_values[column] = (changed_train_values[column] - changed_train_values[column].min()) / (changed_train_values[column].max() - changed_train_values[column].min())
                for column in changed_test_values.columns: 
                    changed_test_values[column] = (changed_test_values[column] - changed_test_values[column].min()) / (changed_test_values[column].max() - changed_test_values[column].min())     
                changed_train_values = changed_train_values.assign(emotion=train_values["emotion"])
                changed_test_values = changed_test_values.assign(emotion=test_values["emotion"])
            case 'rescaled':
                changed_train_values = train_values.copy()
                changed_train_values = changed_train_values.drop("Unnamed: 0",  axis=1)
                changed_test_values = test_values.copy()
                changed_test_values = changed_test_values.drop("Unnamed: 0",  axis=1)
                for column in changed_train_values.columns: 
                    changed_train_values[column] = changed_train_values[column]  / changed_train_values[column].abs().max() 
                for column in changed_test_values.columns: 
                    changed_test_values[column] = changed_test_values[column]  / changed_test_values[column].abs().max()
                changed_train_values = changed_train_values.assign(emotion=train_values["emotion"])
                changed_test_values = changed_test_values.assign(emotion=test_values["emotion"])
            case 'standardized':
                changed_train_values = train_values.copy()
                changed_train_values = changed_train_values.drop("Unnamed: 0",  axis=1)
                changed_test_values = test_values.copy()
                changed_test_values = changed_test_values.drop("Unnamed: 0",  axis=1)
                for column in changed_train_values.columns: 
                    changed_train_values[column] = (changed_train_values[column] - changed_train_values[column].mean()) / changed_train_values[column].std()   
                for column in changed_test_values.columns: 
                    changed_test_values[column] = (changed_test_values[column] - changed_test_values[column].mean()) / changed_test_values[column].std()   
                changed_train_values = changed_train_values.assign(emotion=train_values["emotion"])
                changed_test_values = changed_test_values.assign(emotion=test_values["emotion"])
        changed_train_values.to_csv(os.path.split(train_path)[0] + f"{operation}_train_values.csv")
        changed_test_values.to_csv(os.path.split(test_path)[0] + f"{operation}_test_values.csv")


def calculate_least_important_features(feature_dataset, operation):
    """
    Use Random Forest classification on dataset to order the features by
    their importance for classification. The ordered list gets saved to 2 lists
    which inhibit the features thad add up to 50% respective 80% importance.
    These lists are used to return two lists that represent the least important
    features. 
    This function is intended to be used for automatic model training.

        Params:
            feature_dataset (Pandas.Dataframe): The extracted values from CAER video dataset
            operation (String): The normalization operation that was used on current dataset
        Returns:
            features_to_delete_for_50 (List of int): A list of integers representing the features that make up less than 50% importance
            features_to_delete_for_80 (List of int): A list of integers representing the features that make up less than 20% importance
    """

    print(f"Calculating the importance of features in {operation} dataset. Please wait...")
    X_np_array = np.asarray(feature_dataset.iloc[1:, 1:-1].values, dtype=np.float32)
    y_np_array = np.asarray(feature_dataset.iloc[1:, -1:].values, dtype=np.int8).ravel()
    columns = feature_dataset.columns[1:-1]
    rf = RandomForestClassifier()
    rf.fit(X_np_array, y_np_array)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    sum = 0
    print(f"Feature ranking of  {operation} dataset:")
    over_50 = False
    over_80 = False
    features_to_delete_for_50 = np.arange(len(columns))
    features_to_delete_for_80 = np.arange(len(columns))
    shown_features = []
    for f in range(X_np_array.shape[1]):
        print(f"{f + 1}. Feature {indices[f]}, {columns[indices[f]]}, ({importances[indices[f]]*100} %)")
        sum += importances[indices[f]]
        shown_features.append(indices[f])
        if sum >= 0.5 and not over_50:
            print("50% importance")
            over_50 = True
            features_to_delete_for_50 = np.delete(features_to_delete_for_50, shown_features)
        if sum >= 0.8 and not over_80:
            print("80% importance")
            over_80 = True
            features_to_delete_for_80= np.delete(features_to_delete_for_80, shown_features)
    #print(f"features to delete for 50% importance in {operation} dataset : {features_to_delete_for_50}")
    #print(f"features to delete for 80% importance in {operation} dataset : {features_to_delete_for_80}")
    return features_to_delete_for_50, features_to_delete_for_80


train_and_test_model()