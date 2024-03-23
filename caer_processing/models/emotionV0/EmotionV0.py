from torch import nn, argmax

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
        self.test_dl = 0
        self.train_dl = 0

    def forward(self, x):
        x = x.to(self.fc1.weight.device)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = nn.functional.softmax(x, dim=1)
        return x