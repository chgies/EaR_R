import pandas as pd
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
import EmotionV0

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
    NUM_CLASSES = 7
    NUM_FEATURES = 57
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_file_path = "/###/CAER_CSV/extracted_train_values.csv"  # Replace with the actual path to your CSV file
    test_file_path = "/###/CAER_CSV/extracted_test_values.csv"  # Replace with the actual path to your CSV file
    df_train = pd.read_csv(train_file_path, header=None)
    X_train_np_array = np.asarray(df_train.iloc[1:, 1:-1].values, dtype=np.float32)
    y_train_np_array = np.asarray(df_train.iloc[1:, -1:].values, dtype=np.float32)
    X_train = torch.tensor(X_train_np_array, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train_np_array, dtype=torch.float32).to(device)
    df_test = pd.read_csv(test_file_path, header=None)
    X_test_np_array = np.asarray(df_test.iloc[1:, 1:-1].values, dtype=np.float32)
    y_test_np_array = np.asarray(df_test.iloc[1:, -1:].values, dtype=np.float32)
    X_test = torch.tensor(X_test_np_array, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test_np_array, dtype=torch.float32).to(device)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_dl = DataLoader(train_dataset, batch_size = 12800, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size = 12800, shuffle=True)
    num_of_epochs = 20
    # Put data to target device
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)
    torch.manual_seed(42)
    ### Training
    train_model = EmotionV0(NUM_FEATURES,104,NUM_CLASSES).to(device)
    train_model(train_model, NUM_FEATURES, 104, NUM_CLASSES, train_dl, num_of_epochs)
    # After training the model using train_model function,
    # Evaluate the model on the validation set
    val_model = EmotionV0(NUM_FEATURES,104,NUM_CLASSES).to(device)
    load_model_weights(val_model)
    evaluate_model(val_model, test_dl)

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

def train_model(model, input_size, hidden_size, output_size, data_loader, num_epochs=100, learning_rate=0.001):
    """
    Train the model with the given arguments as parameters
        Params:
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

    # Training loop
    for epoch in range(num_epochs):
        for inputs, labels in data_loader:
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

        # Print loss for every epoch
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    print('Training complete!')

# Function to save the model weights
def save_model_weights(model, filepath='CAER_model_weights.pth'):
    """
    Save the trained weights into a file
        Params:
            model (torch.nn.Module): An instance of the model that has been trained
            filepath (String): The path where the weights are saved
        Returns:
            None
    """
    torch.save(model.state_dict(), filepath)
    print(f'Model weights saved to {filepath}')

# Function to load the model weights
def load_model_weights(model, filepath='CAER_model_weights.pth'):
    """
    Load saved weights into the model
        Params:
            model (torch.nn.Module): An instance of the model that needs these weights
            filepath (String): The path where the weights are saved
    Returns:
        None
    """
    model.load_state_dict(torch.load(filepath))
    print(f'Model weights loaded from {filepath}')

# Function to evaluate the model on a validation set
def evaluate_model(model, val_loader):
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
    print(f'Accuracy on validation set: {accuracy * 100:.2f}%')

train_and_test_model()