from torch import nn, relu, flatten, mean, argmax

class EmotionV2(nn.Module):
    """
    The feature-reduced version of the Neural Net model that gets emotions out of Video data.
    Still a test version!
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(EmotionV2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.test_dl = 0
        self.train_dl = 0

    def forward(self, x):
        x = x.to(self.fc1.weight.device)
        x = self.fc1(x)
        x = self.fc2(x)
        x = relu(x)
        x = self.fc3(x)
        #print(f"x before softmax: {x[0]}")
        #print(f"x.shape before softmax: {x.shape}")
        x = nn.functional.softmax(x, dim=1)
        #print(f"x after softmax: {x[0]}")
        #print(f"x.shape after softmax: {x.shape}")
        return x