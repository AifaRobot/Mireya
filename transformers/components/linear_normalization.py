import torch.nn as nn

"""It is used to apply layer normalization (LayerNorm) to a tensor that is encapsulated in the memory.
Normalization serves a key function: improving learning and model stability."""
class LinearNormalization(nn.Module):

    def __init__(self, normalized_shape):
        super(LinearNormalization, self).__init__()

        self.linear_normalization = nn.LayerNorm(normalized_shape)

    def forward(self, x, _):
        return self.linear_normalization(x)