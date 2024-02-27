import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

def prepare_training():
    NUM_CLASSES = 1
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

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class EmotionV0(nn.Module):
    """
    The first version of the Neural Net model that gets emotions out of Video data.
    Still a test version!
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(EmotionV0, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.to(self.fc1.weight.device)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Function to convert integer labels to one-hot encoded vectors
def convert_to_one_hot(labels, num_classes):
    """
    Convert the "emotion" column into 7 columns for model training
    """
    one_hot = torch.nn.functional.one_hot(labels - 1, num_classes).float()
    return one_hot.view(-1, num_classes)

def train_model(input_size, hidden_size, output_size, data_loader, num_epochs=100, learning_rate=0.001):
    """
    Train the model
    """
    # Initialize the loss function, and optimizer
    model = EmotionV0(input_size,hidden_size, output_size)
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
    

"""
# Make predictions with the model
untrained_preds = model_0(X_test.to(device))
print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(y_test)}, Shape: {y_test.shape}")
print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")
print(f"\nFirst 10 test labels:\n{y_test[:10]}")
"""

# Fit the model
torch.manual_seed(42)

# Set number of epochs
epochs = 20


# Put data to target device
#X_train, y_train = X_train.to(device), y_train.to(device)
#X_test, y_test = X_test.to(device), y_test.to(device)

### Training
#train_model(57, 104, 7, train_dl, epochs)

# Function to save the model weights
def save_model_weights(model, filepath='./CAER_model_weights.pth'):
    torch.save(model.state_dict(), filepath)
    print(f'Model weights saved to {filepath}')

# Function to load the model weights
def load_model_weights(model, filepath='./CAER_model_weights.pth'):
    model.load_state_dict(torch.load(filepath))
    print(f'Model weights loaded from {filepath}')

# Function to evaluate the model on a validation set
def evaluate_model(model, val_loader):
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

# After training the model using train_model function,
# Evaluate the model on the validation set
val_model = EmotionV0(57,104,7).to(device)
load_model_weights(val_model)
evaluate_model(val_model, test_dl)