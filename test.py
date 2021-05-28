import dlc_practical_prologue as prologue
from network import NaiveNet, SharedWeightNet, BenchmarkNet
from helpers import train_model, compute_accuracy

num_samples = 1000

# auxiliary loss parameter
lambda_ = .23  

(train_input, train_target, train_classes,
 test_input, test_target, test_classes) = prologue.generate_pair_sets(num_samples)

# We test four types of nets
titles = [
    'Naive network, no weight sharing, 1 hidden layer',
    'Siamese network with weight sharing, 1 hidden layer',
    'Siamese network with weight sharing, 1 hidden layer and auxiliary loss',
    'Siamese network with weight sharing, 2 hidden layers and auxiliary loss',
    'Siamese network with weight sharing, 2 hidden layers, auxiliary loss and <= operator'
]
models = [NaiveNet(), SharedWeightNet(), SharedWeightNet(), SharedWeightNet(hidden_layers=2), BenchmarkNet(hidden_layers=2)]
hyper_parameters=[
{'learning_rate': 0.0002, 'batch_size': 50, 'lambda_': 0},
{'learning_rate': 0.0005, 'batch_size': 50, 'lambda_': 0},
{'learning_rate': 3.1e-3, 'batch_size': 100, 'lambda_': 0.23},
{'learning_rate': 1.2e-2, 'batch_size': 100, 'lambda_': 0.23},
{'learning_rate': 1.2e-2, 'batch_size': 100, 'lambda_': 0.23}
]

with_aux_loss = [False, False, True, True, True]

for title, model, hp, auxiliary_loss in zip(titles, models, hyper_parameters, with_aux_loss):
    print(title)
    # Train the model using the default number of epochs and batch size
    train_model(model, train_input, train_target, train_classes, learning_rate=hp['learning_rate'], lambda_=hp['lambda_'], batch_size=hp['batch_size'],
        auxiliary_loss=auxiliary_loss)

    # Compute train and test accuracy
    train_accuracy, _ = compute_accuracy(model, train_input, train_target)
    test_accuracy, _ = compute_accuracy(model, test_input, test_target)

    print(
        'Training accuracy: {:.2f}%\nTest accuracy:     {:.2f}%\n'
        .format(train_accuracy * 100, test_accuracy * 100)
    )
    print('#######################\n')
