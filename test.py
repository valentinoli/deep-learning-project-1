import sys
import dlc_practical_prologue as prologue
from network import NaiveNet, SharedWeightNet, BenchmarkNet
from helpers import train_model, compute_accuracy

num_samples = 1000


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

agg_train_results = {t:[] for t in titles}
agg_test_results = {t:[] for t in titles}

for i in range(10):
    train_input, train_target, train_classes, \
    test_input, test_target, test_classes = prologue.generate_pair_sets(num_samples)

    # We test five types of nets with appropriate hyper-parameters

    for title, model, params, auxiliary_loss in zip(titles, models, hyper_parameters, with_aux_loss):
        print(title)
        for k, v in params.items():
            print(f'{k}: {v}')

        # Train the model using the default number of epochs and batch size
        train_model(model, train_input, train_target, train_classes, **params, auxiliary_loss=auxiliary_loss)

        sys.stdout.write('\rTraining complete!\n')

        # Compute train and test accuracy
        train_accuracy = compute_accuracy(model, train_input, train_target)
        test_accuracy = compute_accuracy(model, test_input, test_target)
        
        agg_train_results[title].append(train_accuracy)
        agg_test_results[title].append(test_accuracy)

        
        

print(
    'Training accuracy: {:.2f}%\nTest accuracy:     {:.2f}%\n'
    .format(train_accuracy * 100, test_accuracy * 100)
)
print('#######################\n')
