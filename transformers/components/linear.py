import torch.nn as nn

"""It's a fully connected layer (also known as a dense layer). The linear layer combines information
from all the input features and allows the network to generate more complex representations."""
class Linear(nn.Module):

    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, _):
        return self.linear(x)