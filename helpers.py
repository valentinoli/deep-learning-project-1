from __future__ import annotations
import sys

import torch
from torch import nn, Tensor
from network import NaiveNet, SharedWeightNet, BenchmarkNet
import dlc_practical_prologue as prologue

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


def k_fold_split(inputs: Tensors, folds: int = 4):
    """
    :param inputs: input tensor of dimension (N, 2, 14, 14), N is the number of pairs
    :param folds: number of inputs split for cross-validation
    :returns: indices of the entries corresponding to each fold
    """

    shuffle_indices = torch.randperm(len(inputs))
    split_indices = torch.split(shuffle_indices, int(torch.tensor(len(inputs) / folds).item()))

    kfold_train = []
    kfold_valid = []

    for i in range(folds):
        kfold_train_indices = torch.cat(split_indices[0:i] + split_indices[i+1:])
        kfold_validation_indices = split_indices[i]

        kfold_train.append(kfold_train_indices)
        kfold_valid.append(kfold_validation_indices)

    return kfold_train, kfold_valid




def grid_search(
    constructor: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    classes: Tensor,
    layers: int,
    auxiliary_loss: bool,
    params: list,
    folds: int = 4,
    verbose: bool = True):
    """
    :param model: the model
    :param inputs: input tensor of dimension (N, 2, 14, 14), N is the number of pairs
    :param labels: ground truth tensor of dimension N
    :param classes: inputs classes tensor of dimensions (N, 2)
    :param parems: set of hyper-parameters in a dictionnary
    :param folds: number of inputs split for cross-validation
    :returns: number of errors for each set of parameters
    """

    kfold_train_idx, kfold_valid_idx = k_fold_split(inputs, folds)

    train_error = []
    valid_error = []
    
    for val in params:

        kfold_train_error = []
        kfold_valid_error = []

        for i in range(folds):
            model = constructor(hidden_layers=layers)
            train_model(
                model,
                inputs=inputs[kfold_train_idx[i]],
                targets=targets[kfold_train_idx[i]],
                classes=classes[kfold_train_idx[i]],
                learning_rate= val['lr'],
                lambda_= val['lambda_'],
                batch_size= val['batch_size'],
                nb_epochs = 25,
                auxiliary_loss=auxiliary_loss,
                verbose=verbose)

            kfold_train_error.append(compute_accuracy(model, inputs[kfold_train_idx[i]], targets[kfold_train_idx[i]])[1])
            kfold_valid_error.append(compute_accuracy(model, inputs[kfold_valid_idx[i]], targets[kfold_valid_idx[i]])[1])

        train_error.append(torch.tensor(kfold_train_error ,dtype=torch.float).mean())
        valid_error.append(torch.tensor(kfold_valid_error ,dtype=torch.float).mean())

    return train_error, valid_error

def tune_hyperparameters(N=1000):

    model_constructors = [NaiveNet, SharedWeightNet, SharedWeightNet, SharedWeightNet, BenchmarkNet]
    hidden_layers = [1,1,1,2,2]
    with_aux_loss = [False, False, True, True, True]

    params_without_auxi_loss=[]

    for lr in torch.logspace(start=-4, end=-1, steps=5):
        for batch_size in [25, 50, 125]:
            params_without_auxi_loss.append({'lr':lr, 'batch_size':batch_size, 'lambda_':0})

    params_with_auxi_loss=[]

    for lr in torch.logspace(start=-4, end=-1, steps=5):
        for batch_size in [25, 50, 125]:
            for lambda_ in torch.logspace(start=-2, end=0, steps=6)[:-1]:
                params_with_auxi_loss.append({'lr':lr, 'batch_size':batch_size, 'lambda_':lambda_})

    train_input, train_target, train_classes, \
    test_input, test_target, test_classes = \
    prologue.generate_pair_sets(N)
    
    
    params_iterate = [params_without_auxi_loss, params_without_auxi_loss, params_with_auxi_loss, params_with_auxi_loss, params_with_auxi_loss ]
    results = []
    test_error = []

    for (m, hl, aux, p) in zip(model_constructors, hidden_layers, with_aux_loss, params_iterate):
        _, test = grid_search(
                    m,
                    inputs=train_input,
                    targets=train_target,
                    classes=train_classes,
                    layers=hl,
                    auxiliary_loss=aux,                
                    params=p,
                    folds=4,
                    verbose=False)
        test_error.append(test)
        results.append(p[torch.tensor(test, dtype=float).argmin()])
        print(p[torch.tensor(test, dtype=float).argmin()])


    return results, test_error