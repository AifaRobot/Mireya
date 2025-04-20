import torch.nn as nn

"""Allows you to perform a transformation at the input"""
class ShapeTransformation(nn.Module):

    def __init__(self, transformation):
        super(ShapeTransformation, self).__init__()

        self.transformation = transformation

    def forward(self, x, _):
        return self.transformation(x)