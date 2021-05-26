from network import *
import dlc_practical_prologue as prologue

def train_and_test_model(model, N, mini_batch_size, epochs, lambda_, learning_rate, auxiliary_loss=False):
	#generate train/test set
	train_input, train_target, train_classes, \
	test_input, test_target, test_classes = \
	prologue.generate_pair_sets(N)

	if auxiliary_loss:
		#train model using auxiliary loss
		train_model(model, train_input, train_target, train_classes, learn_rate_= learning_rate,
					lambda_=lambda_, mini_batch_size=mini_batch_size, nb_epochs = epochs)
	else:
		train_model(model, train_input, train_target, learn_rate_= learning_rate, lambda_=0,
			mini_batch_size=mini_batch_size, nb_epochs = epochs)

	#count errors
	train_error = compute_nb_errors(model, train_input, train_target, mini_batch_size)
	test_error = compute_nb_errors(model, test_input, test_target, mini_batch_size)
	
	return train_error, test_error

def print_acc(model_name, train_error, test_error, N):
	print(f'{model_name}\n\
\ttraining set accurary : {100-train_error/N*100}%\n\
\ttest set accuracy : {100-test_error/N*100}%\n')

#params
N=1000
epochs=25

#hyper-parameters
mini_batch_size = 100
lambda_ = .23

#Naive network, no weight sharing, 1 hidden layer
Naive = NaiveCNN(1)
Naive_name = "naive network (1 hidden)"
tr_e, te_e = train_and_test_model(Naive, N, mini_batch_size, epochs, lambda_, learning_rate=1.6e-3, auxiliary_loss=False)
print_acc(Naive_name, tr_e, te_e, N)

#Siamese network, 1 hidden layer
Siamese_1h = SharedWeight(1)
Siamese_1h_n = "siamese network (1 hidden)"
tr_e, te_e = train_and_test_model(Siamese_1h, N, mini_batch_size, epochs, lambda_, learning_rate=1.6e-3, auxiliary_loss=False)
print_acc(Siamese_1h_n, tr_e, te_e, N)

#Siamese network, 1 hidden layer with auxi loss
Siamese_1h = SharedWeight(1)
Siamese_1h_n = "siamese network (1 hidden) with auxi. loss"
tr_e, te_e = train_and_test_model(Siamese_1h, N, mini_batch_size, epochs, lambda_, learning_rate=3.1e-3, auxiliary_loss=True)
print_acc(Siamese_1h_n, tr_e, te_e, N)

#Siamese network, 2 hidden layer
Siamese_2h = SharedWeight(2)
Siamese_2h_n = "siamese network (2 hidden) with auxi. loss"
tr_e, te_e = train_and_test_model(Siamese_2h, N, mini_batch_size, epochs, lambda_, learning_rate=1.2e-2, auxiliary_loss=True)
print_acc(Siamese_2h_n, tr_e, te_e, N)
