import sys
import dlc_practical_prologue as prologue
from network import NaiveNet, SharedWeightNet, BenchmarkNet
from helpers import train_model, compute_accuracy, bootstrapped_std

num_samples = 1000

# We test five types of nets with appropriate hyper-parameters
titles = [
    'Naive network, no weight sharing, 1 hidden layer',
    'Siamese network with weight sharing, 1 hidden layer',
    'Siamese network with weight sharing, 1 hidden layer and auxiliary loss',
    'Siamese network with weight sharing, 2 hidden layers and auxiliary loss',
    'Siamese network with weight sharing, 2 hidden layers, auxiliary loss and <= operator'
]

models = [
    NaiveNet(),
    SharedWeightNet(),
    SharedWeightNet(),
    SharedWeightNet(hidden_layers=2),
    BenchmarkNet(hidden_layers=2)
]

hyper_parameters = [
    {'learning_rate': 0.0002, 'batch_size': 50, 'lambda_': 0},
    {'learning_rate': 0.0005, 'batch_size': 50, 'lambda_': 0},
    {'learning_rate': 3.1e-3, 'batch_size': 100, 'lambda_': 0.23},
    {'learning_rate': 1.2e-2, 'batch_size': 100, 'lambda_': 0.23},
    {'learning_rate': 1.2e-2, 'batch_size': 100, 'lambda_': 0.23}
]

with_aux_loss = [False, False, True, True, True]

for title, model, params, auxiliary_loss in zip(titles, models, hyper_parameters, with_aux_loss):
    print(title)

    for k, v in params.items():
        print(f'{k}: {v}')

    train_accuracy = []
    test_accuracy = []

    for i in range(2):

        train_input, train_target, train_classes, \
        test_input, test_target, test_classes = prologue.generate_pair_sets(num_samples)
        
        # Train the model using the default number of epochs and batch size
        train_model(model, train_input, train_target, train_classes, **params, auxiliary_loss=auxiliary_loss)
    
        # Compute train and test accuracy
        train_accuracy.append(compute_accuracy(model, train_input, train_target))
        test_accuracy.append(compute_accuracy(model, test_input, test_target))

    sys.stdout.write('\rTraining complete!\n')

    #compute boostrapped std and average
    train_avg, train_std = bootstrapped_std(train_accuracy,1000)
    test_avg, test_std = bootstrapped_std(test_accuracy,1000)


    print(
        'Training accuracy: {:.2f}±{:.2f}%\nTest accuracy:     {:.2f}±{:.2f}%\n'
        .format(train_avg * 100, train_std*100,
                test_avg * 100, test_std*100)
    )
    print('#######################\n')
