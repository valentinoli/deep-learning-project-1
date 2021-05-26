from network import *
import dlc_practical_prologue as prologue

#params
N=1000
epochs=25

#hyper-parameters
mini_batch_size = 100
learning_rate = 2.2e-2
lambda_ = .23

#generate train/test set
train_input, train_target, train_classes, \
test_input, test_target, test_classes = \
	prologue.generate_pair_sets(N)

#declare model, 2 hidden layer
Siamese = SharedWeight(2)

#train model using auxiliary loss
train_model(Siamese, train_input, train_target, train_classes, learn_rate_= learning_rate, lambda_=lambda_,
			mini_batch_size=mini_batch_size, nb_epochs = epochs)

#
train_error = compute_nb_errors(Siamese, train_input, train_target, mini_batch_size)
test_error = compute_nb_errors(Siamese, test_input, test_target, mini_batch_size)

print(f'The model obtained a {100-train_error/N*100}% accuracy on the training set and {100-test_error/N*100}% accuracy on the test set')