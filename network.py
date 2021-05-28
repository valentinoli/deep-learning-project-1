from __future__ import annotations

import torch
from torch import nn, Tensor
from network_helpers import block_cnn, block_digit_classifier, block_output, leq


class Net(nn.Module):
    def __init__(self, hidden_layers: int):
        super().__init__()
        # CNN block
        self.cnn = block_cnn()
        
        # Classification blocks
        self.dc1 = block_digit_classifier(hidden_layers)
        self.dc2 = block_digit_classifier(hidden_layers)
        
        # Output block
        self.out = block_output()
        
    def forward(self, input_: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError


class NaiveNet(Net):
    """Naive implementation without shared weights"""
    def __init__(self, hidden_layers: int = 1):
        super().__init__(hidden_layers)

    def forward(self, input_):
        # Pass each image of every pair through the convolutional block
        output1 = self.cnn(input_.select(1, 0).unsqueeze(1))
        output2 = self.cnn(input_.select(1, 1).unsqueeze(1))
        
        # Predict implicitly the classes of the images, using 2 different networks
        # -> one network for the first image of each pair, and another for the second image
        output1 = self.dc1(output1.flatten(1))
        output2 = self.dc2(output2.flatten(1))
        
        # Predict the output from concatenated class predictions
        output = self.out(torch.cat((output1, output2), 1))
        return output, output1, output2


class SharedWeightNet(Net):
    """Network with shared weights"""
    def __init__(self, hidden_layers: int = 1):
        super().__init__(hidden_layers)

    def forward_one(self, input_: Tensor):
        """Forwards the one image of each pair through the CNN and classification blocks"""
        return self.dc1(self.cnn(input_).flatten(1))
    
    def forward_both(self, input_: Tensor):
        """Forwards both images through the shared weight net"""
        output1 = self.forward_one(input_.select(1, 0).unsqueeze(1))
        output2 = self.forward_one(input_.select(1, 1).unsqueeze(1))
        return output1, output2
    
    def forward(self, input_):
        """Forward pass of the input batch of image pairs"""
        # Forward both images in every pair through the first two blocks
        # -> shared weight
        output1, output2 = self.forward_both(input_)
        
        # Computes the final output by concatenating the intermediate outputs of each pair 
        output = self.out(torch.cat((output1, output2), 1))
        return output, output1, output2


class BenchmarkNet(SharedWeightNet):
    """Network with shared weights with simple boolean operator for output prediction"""
    def __init__(self, hidden_layers: int = 2):
        super().__init__(hidden_layers)
        
    def forward(self, input_):
        """Forward pass of the input batch of image pairs"""
        output1, output2 = self.forward_both(input_)
        
        # Compute the output binary distribution by directly computing
        # if the most likely digit 1 is lesser or equal to the most likely digit 2
        output = leq(output1.argmax(1), output2.argmax(1))
        return output, output1, output2
