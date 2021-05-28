import sys
from collections import defaultdict

import dlc_practical_prologue as prologue
from network import NaiveNet, SharedWeightNet, BenchmarkNet
from helpers import train_model, compute_accuracy, bootstrapping

num_samples = 1000

# Change this parameter to control the number of rounds
NUM_ROUNDS = 3

# We test five types of nets with appropriate hyper-parameters
titles = [
    'Naive network, no weight sharing, 1 hidden layer',
    'Siamese network with weight sharing, 1 hidden layer',
    'Siamese network with weight sharing, 1 hidden layer and auxiliary loss',
    'Siamese network with weight sharing, 2 hidden layers and auxiliary loss',
    'Siamese network with weight sharing, 2 hidden layers, auxiliary loss and <= operator'
]

hyper_parameters = [
    {'learning_rate': 0.0002, 'batch_size': 100, 'lambda_': 0},
    {'learning_rate': 0.0005, 'batch_size': 100, 'lambda_': 0},
    {'learning_rate': 3.1e-3, 'batch_size': 100, 'lambda_': 0.23},
    {'learning_rate': 1.2e-2, 'batch_size': 100, 'lambda_': 0.23},
    {'learning_rate': 1.2e-2, 'batch_size': 100, 'lambda_': 0.23}
]

with_aux_loss = [False, False, True, True, True]

# dicts to collect train and test errors for each model
agg_train_results = defaultdict(list)
agg_test_results = defaultdict(list)

print(f'Running {NUM_ROUNDS} rounds of {len(titles)} models.')
print('Change NUM_ROUNDS to control the number of rounds\n')

for i in range(NUM_ROUNDS):
    # reset models in each round
    models = [
        NaiveNet(),
        SharedWeightNet(),
        SharedWeightNet(),
        SharedWeightNet(hidden_layers=2),
        BenchmarkNet(hidden_layers=2)
    ]
    print(f'\n*** Round {i+1} ***\n')
    train_input, train_target, train_classes, \
    test_input, test_target, test_classes = prologue.generate_pair_sets(num_samples)

    for title, model, params, auxiliary_loss in zip(titles, models, hyper_parameters, with_aux_loss):
        print(title)
        
        # Train the model using the default number of epochs and batch size
        train_model(model, train_input, train_target, train_classes, **params, auxiliary_loss=auxiliary_loss)

        sys.stdout.write('\rTraining complete!\n')

        # Compute train and test accuracy
        train_accuracy = compute_accuracy(model, train_input, train_target)
        test_accuracy = compute_accuracy(model, test_input, test_target)
        
        agg_train_results[title].append(train_accuracy)
        agg_test_results[title].append(test_accuracy)

    sys.stdout.write(f'\rRound {i+1} finished for all models!\n')

print('\n\n#######################\n\n')

for t, p in zip(titles, hyper_parameters):
    print('\n',t)
    for k, v in p.items():
        print(f'{k}: {v}')

    train_errors = agg_train_results[t]
    test_errors = agg_test_results[t]
    
    train_mean, train_std = bootstrapping(train_errors, bootstrap=1000)
    test_mean, test_std = bootstrapping(test_errors, bootstrap=1000)
    
    print('Training accuracy: {:.2f}% ± {:.2f}%'.format(train_mean * 100, train_std * 100))
    print('Test accuracy:     {:.2f}% ± {:.2f}%'.format(test_mean * 100, test_std * 100))
