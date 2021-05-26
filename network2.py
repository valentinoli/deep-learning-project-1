import torch
from torch import nn
from network_helpers import block_cnn, block_digit_classifier, block_output

class NaiveCNN(nn.Module):
    """Naive implementation of a deep CNN classifier"""
    def __init__(self, hidden_layers: int = 2):
        super().__init__()        
        self.cnn = block_cnn()
        
        self.dc1 = block_digit_classifier(hidden_layers)
        self.dc2 = block_digit_classifier(hidden_layers)
        
        self.out = block_output()

    def forward(self, input_):
        # call the network on both inputs
        output1 = self.cnn(input_.select(1, 0).unsqueeze(1))
        output2 = self.cnn(input_.select(1, 1).unsqueeze(1))
        
        # Predict implicitly the classes of input1 and input2, using 2 different networks
        output1 = self.dc1(output1.flatten(1))
        output2 = self.dc2(output2.flatten(1))
        
        # Predict the output from concatenated class predictions
        output = self.out(torch.cat((output1, output2), 1))
        return output, output1, output2


    
class SharedWeight(nn.Module):
    def __init__(self, hidden_layers: int = 2):
        super().__init__()
        self.cnn = block_cnn()
        
        # shared weight block
        self.dc = block_digit_classifier(hidden_layers)
        
        self.out = block_output()

    def forward_one(self, x):
        """Forwards the one image of each pair through the CNN and classification blocks"""
        return self.dc(self.cnn(x).flatten(1))
    
    def forward(self, input_):
        """Forward pass of the input batch of image pairs"""
        # Call the first two blocks on both images
        output1 = self.forward_one(input_.select(1, 0).unsqueeze(1))
        output2 = self.forward_one(input_.select(1, 1).unsqueeze(1))
        
        # Computes the final output by concatenating the intermediate outputs of each pair 
        output = self.out(torch.cat((output1, output2), 1))
        return output, output1, output2
    