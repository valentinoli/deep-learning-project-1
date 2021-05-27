import torch
from torch import nn
from network_helpers import block_cnn, block_digit_classifier, block_output, leq

class NaiveNet(nn.Module):
    """Naive implementation without shared weights"""
    def __init__(self, hidden_layers: int = 1):
        super().__init__()        
        self.cnn = block_cnn()
        
        self.dc1 = block_digit_classifier(hidden_layers)
        self.dc2 = block_digit_classifier(hidden_layers)
        
        self.out = block_output()

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


class SharedWeightNet(nn.Module):
    """Network with shared weights"""
    def __init__(self, hidden_layers: int = 1):
        super().__init__()
        self.cnn = block_cnn()
        
        # Shared weight block
        self.dc = block_digit_classifier(hidden_layers)
        
        self.out = block_output()

    def forward_one(self, x):
        """Forwards the one image of each pair through the CNN and classification blocks"""
        return self.dc(self.cnn(x).flatten(1))
    
    def forward(self, input_):
        """Forward pass of the input batch of image pairs"""
        # Forward both images in every pair through the first two blocks
        # -> shared weight
        output1 = self.forward_one(input_.select(1, 0).unsqueeze(1))
        output2 = self.forward_one(input_.select(1, 1).unsqueeze(1))
        
        # Computes the final output by concatenating the intermediate outputs of each pair 
        output = self.out(torch.cat((output1, output2), 1))
        return output, output1, output2

class BenchmarkNet(nn.Module):
    """Network with shared weights with simple boolean operator for output prediction"""
    def __init__(self, hidden_layers: int = 2):
        super().__init__()
        self.cnn = block_cnn()
        
        # shared weight block
        self.dc = block_digit_classifier(hidden_layers)
        
    def forward_one(self, x):
        """Forwards the one image of each pair through the CNN and classification blocks"""
        return self.dc(self.cnn(x).flatten(1))
    
    def forward(self, input_):
        """Forward pass of the input batch of image pairs"""
        # Call the first two blocks on both images
        output1 = self.forward_one(input_.select(1, 0).unsqueeze(1))
        output2 = self.forward_one(input_.select(1, 1).unsqueeze(1))
        
        # Computes the final output by concatenating the intermediate outputs of each pair 
        output = leq(output1.argmax(1), output2.argmax(1))
        return output, output1, output2