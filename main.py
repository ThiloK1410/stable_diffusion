import torch
import torch.nn as nn
import torch.nn.functional as F

class NET(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)

        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)

        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Pass data through conv1
        x = self.conv1(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        # Run max pooling over x
        x = F.max_pool2d(x, 2)
        # Pass data through dropout1
        x = self.dropout1(x)
        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)
        # Pass data through ``fc1``
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        # Apply softmax to x
        output = F.log_softmax(x, dim=1)
        return output


class DropoutTest(nn.Module):
    def __init__(self):
        super().__init__()

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(x)

        return x


if __name__ == "__main__":
    my_nn = NET()
    dropout_test = DropoutTest()

    random_data = torch.rand((1, 1, 28, 28))

    result = my_nn(random_data)
    print(result)
