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
    epochs: int = 25,
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

    for epoch in range(epochs):
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


@torch.no_grad()
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


def compute_num_correct(predictions: Tensor, targets: Tensor) -> int:
    """
    :param predictions: see predict() return value
    :param targets: ground truth target values
    :returns: number of correct predictions
    """
    return (predictions == targets).sum()


def compute_accuracy(model: nn.Module, inputs: Tensor, targets: Tensor) -> float:
    """
    :param model: the model
    :param inputs: input tensor of dimension (N, 2, 14, 14), N is the number of pairs
    :param targets: ground truth tensor of dimension N
    :returns: total accuracy
    """
    num_correct = compute_num_correct(predict(model, inputs), targets)
    accuracy = num_correct / len(targets)
    return accuracy


def compute_num_errors(model: nn.Module, inputs: Tensor, targets: Tensor) -> int:
    """
    :param model: the model
    :param inputs: input tensor of dimension (N, 2, 14, 14), N is the number of pairs
    :param targets: ground truth tensor of dimension N
    :returns: number of incorrect predictions
    """
    num_correct = compute_num_correct(predict(model, inputs), targets)
    num_errors = len(targets) - num_correct
    return num_errors


def bootstrapping(values: list, size: int = 500) -> Tensor:
    """
    :param values: list of values to bootstrap resample from
    :param size: number of bootstrap samples
    :returns: tensor of resampled values
    """
    values = torch.tensor(values, dtype=torch.float)
    bootstrap_sample_indices = torch.randint(len(values), (size,))
    return values[bootstrap_sample_indices]


def k_fold_split(num_samples: int, k: int = 4) -> tuple[Tensor]:
    """
    :param num_samples: number of samples
    :param k: number of folds to split the dataset for cross-validation
    :returns: k folds of indices
    """
    fold_size = int(num_samples / k)
    indices_shuffled = torch.randperm(num_samples)
    indices_split = torch.split(indices_shuffled, fold_size)
    return indices_split


def grid_search(
    constructor,
    inputs: Tensor,
    targets: Tensor,
    classes: Tensor,
    params: list[dict],
    hidden_layers: int,
    auxiliary_loss: bool = False,
    epochs: int = 25,
    k: int = 4,
    verbose: bool = True,
) -> tuple[list, list]:
    """
    :param model: the model
    :param inputs: input tensor of dimension (N, 2, 14, 14), N is the number of pairs
    :param labels: ground truth tensor of dimension N
    :param classes: inputs classes tensor of dimensions (N, 2)
    :param params: set of combination of hyper-parameters
    :param hidden_layers: number of hidden layers in the model
    :param auxiliary_loss: whether to train with auxiliary loss
    :param epochs: number of epochs
    :param k: number of folds for the cross-validation
    :param verbose: switch for verbose output
    :returns: number of errors for each set of parameters
    """
    kfolds = k_fold_split(len(inputs), k=k)

    train_error = []
    valid_error = []
    
    for param_dict in params:
        print(f'Params {param_dict}')
        kfold_train_error = []
        kfold_valid_error = []

        for i in range(k):
            print(f'Fold {i+1}')
            # create new model
            model = constructor(hidden_layers=hidden_layers)
            
            # compute train indices and validation indices
            train_idx = torch.cat(kfolds[:i] + kfolds[i+1:])
            valid_idx = kfolds[i]
            
            # train the model on the train set
            train_model(
                model,
                inputs=inputs[train_idx],
                targets=targets[train_idx],
                classes=classes[train_idx],
                epochs=epochs,
                auxiliary_loss=auxiliary_loss,
                verbose=verbose,
                **param_dict
            )
            
            # keep track of validation errors in each fold
            kfold_train_error.append(compute_num_errors(model, inputs[train_idx], targets[train_idx]))
            kfold_valid_error.append(compute_num_errors(model, inputs[valid_idx], targets[valid_idx]))

        # compute average error over the folds
        train_error.append(torch.tensor(kfold_train_error, dtype=torch.float).mean())
        valid_error.append(torch.tensor(kfold_valid_error, dtype=torch.float).mean())

    return train_error, valid_error


def tune_hyperparameters(num_samples = 1000, k = 4, verbose = False, start = 0, end = 5):
    """
    Tune hyperparameters of all the models using grid search
    :param num_samples: number of samples
    :param k: number of folds for cross validation
    :param start: start model index
    :param end: end model index (excluded)
    :returns: best parameters, training and test errors, parameters
    """
    model_constructors = [NaiveNet, SharedWeightNet, SharedWeightNet, SharedWeightNet, BenchmarkNet][start:end]
    hidden_layers = [1, 1, 1, 2, 2][start:end]
    with_aux_loss = [False, False, True, True, True][start:end]
    
    learning_rates = [0.0002, 0.0005, 0.0031, 0.012, 0.1]
    lambdas = [0, 0.115, 0.23, 0.345]

    params_without_auxi_loss = []
    params_with_auxi_loss = []

    for lr in learning_rates:
        p = {'learning_rate': lr, 'batch_size': 100}
        params_without_auxi_loss.append(p)
        for lambda_ in lambdas:
            params_with_auxi_loss.append({**p, 'lambda_': lambda_})

    inputs, targets, classes, _, _, _ = prologue.generate_pair_sets(num_samples)

    params_dict = {
        False: params_without_auxi_loss,
        True: params_with_auxi_loss
    }

    results = []
    test_error = []
    params_iterate = []
    train_error = []
    
    for m, hl, aux in zip(model_constructors, hidden_layers, with_aux_loss):
        print(f'Performing hyperparameter-tuning on {m.__name__} with {hl} hidden layers')
        print(f'Auxiliary loss: {aux}')
        params = params_dict[aux]
        
        train, test = grid_search(
            m,
            inputs,
            targets,
            classes,
            hidden_layers=hl,
            auxiliary_loss=aux,                
            params=params,
            k=k,
            verbose=False
        )
        test_error.append(test)
        train_error.append(train)
        params_iterate.append(params)
        best_params = params[torch.tensor(test, dtype=float).argmin()]
        results.append(best_params)
        print(best_params)
        
        if verbose:
            print(test, m(hidden_layers=hl))

    return results, train_error, test_error, params_iterate
