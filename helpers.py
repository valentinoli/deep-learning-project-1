from __future__ import annotations
import sys
from typing import Optional

import torch
from torch import nn, Tensor

def create_dataloader(*tensors, batch_size = 10, shuffle = True):
    """Creates a PyTorch data loader from the given tensors"""
    dataset = torch.utils.data.TensorDataset(*tensors)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def train_model(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    classes: Tensor,
    learning_rate: float = 1e-2,
    lambda_: float = 0.1,
    batch_size: int = 100,
    nb_epochs: int = 25,
    auxiliary_loss: bool = False,
    verbose: bool = True
):
    """
    Trains the given model with SGD with the given number of epochs and batch size
    """
    # Create a data loader to iterate over minibatches
    loader = create_dataloader(inputs, targets, classes, batch_size=batch_size)
    
    # We use cross entropy loss, suitable for the binary classification task
    criterion = nn.CrossEntropyLoss()
    
    # SGD optimizer
    optim = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(nb_epochs):
        for inputs_, targets_, classes_ in loader:
            # Forward pass
            # -> returns the binary output as well as 10 class predictions for each image
            output, class_0, class_1 = model(inputs_)
            
            loss = criterion(output, targets_)
            if auxiliary_loss:
                class_0_target = classes_.select(1, 0)
                class_1_target = classes_.select(1, 1)
                loss += lambda_ * (criterion(class_0, class_0_target) + criterion(class_1, class_1_target))
            
            optim.zero_grad()
            
            # Back propagation
            loss.backward()
            
            # SGD step
            optim.step()
    
        if verbose:
            sys.stdout.write(f'\rEpoch {epoch+1}')
            sys.stdout.flush()
    sys.stdout.write('\rTraining complete!\n')


            
def predict(model: nn.Module, inputs: Tensor) -> Tensor:
    """
    :param model: the model
    :param inputs: input tensor of dimension (N, 2, 14, 14), N is the number of pairs
    :returns: binary predictions of the model given the inputs
    """
    # Pass the inputs through the model and get as output a tensor
    # of dimension (N, 2). For each pair we have
    # a probability distribution over 2 binary outputs
    outputs, _, _ = model(inputs)
    
    # We take the argmax of the two probabilities for each pair
    predictions = outputs.argmax(dim=1)
    return predictions


def compute_accuracy(model: nn.Module, inputs: Tensor, targets: Tensor):
    """
    :param model: the model
    :param inputs: input tensor of dimension (N, 2, 14, 14), N is the number of pairs
    :param labels: ground truth tensor of dimension N
    :returns: total accuracy along with correctness tensor and predictions
    """
    predictions = predict(model, inputs)
    correct_class = predictions == targets  # -> boolean tensor
    num_correct = correct_class.sum()
    num_errors = len(targets) - num_correct
    accuracy = num_correct / len(targets)
    return accuracy, num_errors
