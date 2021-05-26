from torch import nn

"""Layer helpers"""
def conv(in_channels, out_channels, kernel_size = 3, padding = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

def pool(kernel_size = 2, stride = 2):
    return nn.MaxPool2d(2, stride=stride)

def linear(in_features, out_features):
    return nn.Linear(in_features, out_features)

def relu(inplace = True):
    return nn.ReLU(inplace=inplace)

def softmax(dim = 1):
    return nn.Softmax(dim)

def sequential(*modules):
    return nn.Sequential(*modules)


"""Block helpers"""

def block_cnn():
    """
    Returns a convolutional block with max pooling to learn meaningful representations in MNIST images
    input dimension: 1 x 14 x 14
    output dimension: 64 x 3 x 3
    """
    return sequential(
        # 1 x 14 x 14
        # -> 32 x 14 x 14
        conv(1, 32),
        # -> 32 x 7 x 7
        pool(),
        # -> 64 x 7 x 7
        conv(32, 64),
        # -> 64 x 3 x 3
        pool()
    )


def block_digit_classifier(hidden_layers):
    """
    Returns a fully-connected block for MNIST digit classification
    input units: 64 * 3 * 3 = 576
    output units: 10
    """
    modules = [
        linear(64 * 3 * 3, 100),
        relu()
    ]

    if hidden_layers == 1:
        modules.extend([
            linear(100, 50),
            relu()
        ])
    elif hidden_layers == 2:
        modules.extend([
            linear(100, 70),
            relu(),
            linear(70, 50),
            relu()
        ])
    else:
        raise ValueError('Invalid number of hidden layers')

    # 10 output units, one for each digit classification
    modules.append(linear(50, 10))
    return sequential(*modules)


def block_output():
    """
    Returns a fully-connected block that synthesizes the classification results
    of the two digits and outputs 
    input units: 20
    output units: 1
    """
    return sequential(
        linear(20, 10),
        relu(),
        linear(10, 2),
        softmax()
    )
