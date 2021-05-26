import torch
from torch import nn

class NaiveCNN(nn.Module):
    def __init__(self, hidden_layer=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2))
             
        #predict implicitely the class of input1 and input2 separately
        classif=[nn.Linear(64*3*3, 100),
                 nn.ReLU(inplace=True)]
        
        if hidden_layer == 1:
            classif.append(nn.Linear(100, 50))
            classif.append(nn.ReLU(inplace=True))
        elif hidden_layer == 2:
            classif.append(nn.Linear(100, 70))
            classif.append(nn.ReLU(inplace=True))
            classif.append(nn.Linear(70, 50))
            classif.append(nn.ReLU(inplace=True))

        classif.append(nn.Linear(50, 10))
        
        self.fc1 = nn.Sequential(*classif)
        self.fc2 = nn.Sequential(*classif)
        
        #predict the output from concatenated classes predictions
        self.fc3 = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10,2),
            nn.Softmax(1))
            
    def forward(self, input_):
        #call the network on both input
        output1 = self.cnn(input_[:,0,:].unsqueeze(1))
        output2 = self.cnn(input_[:,0,:].unsqueeze(1))
        
        #reshape
        output1 = output1.view(output1.size()[0], -1)
        output2 = output2.view(output2.size()[0], -1)
        
        #predict class implicitely, using 2 different network
        output1 = self.fc1(output1)
        output2 = self.fc2(output2)
        
        #predict binary output
        output = self.fc3(torch.cat((output1, output2), 1))
        return output, output1, output2
    
class SharedWeight(nn.Module):
    def __init__(self, hidden_layer=0):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2))
        
        #predict implicitely the class of both image
        classif=[nn.Linear(64*3*3, 100),
                 nn.ReLU(inplace=True)]
        
        if hidden_layer == 1:
            classif.append(nn.Linear(100, 50))
            classif.append(nn.ReLU(inplace=True))
        elif hidden_layer == 2:
            classif.append(nn.Linear(100, 70))
            classif.append(nn.ReLU(inplace=True))
            classif.append(nn.Linear(70, 50))
            classif.append(nn.ReLU(inplace=True))
            
        classif.append(nn.Linear(50, 10))
        self.fc1 = nn.Sequential(*classif)
        
        #predict the output from concatenated classes predictions
        self.fc2 = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10,2),
            nn.Softmax(1))
            
    def forward_once(self, x):
        #call the classification block on an individual input
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output
    
    def forward(self, input_):
        #call the network on both input, combine the outputs and final output
        output1 = self.forward_once(input_[:,0,:].unsqueeze(1))
        output2 = self.forward_once(input_[:,1,:].unsqueeze(1))
        output = self.fc2(torch.cat((output1, output2), 1))
        return output, output1, output2

def train_model(model, input_, target_, classes_=None, learn_rate_= 1e-2, lambda_=0.1, mini_batch_size=100, nb_epochs = 25):
    #classification & binary output
    criterion = nn.CrossEntropyLoss()
    
    #stochastic gradient descent
    optim = torch.optim.SGD(model.parameters(), lr=learn_rate_, momentum=0.9)

    for e in range(nb_epochs):
        for b in range(0, input_.size(0), mini_batch_size):
            #split the input in two, return the binary value and the classes of both input
            output, class_0, class_1 = model(input_.narrow(0, b, mini_batch_size))
            
            if classes_ is None:
                #no auxiliary loss
                loss = criterion(output, target_.narrow(0, b, mini_batch_size))
            else:
                #auxiliary loss
                loss = criterion(output, target_.narrow(0, b, mini_batch_size)) \
                        + lambda_*criterion(class_0, classes_.narrow(0, b, mini_batch_size)[:,0]) \
                        + lambda_*criterion(class_1, classes_.narrow(0, b, mini_batch_size)[:,1])
            
            #backprop
            optim.zero_grad()
            loss.backward()
            optim.step()
    return loss
        
def compute_nb_errors(model, input_, target_, mini_batch_size):
    nb_error = 0
    
    for b in range(0, input_.size(0), mini_batch_size):
        output, _, _ = model(input_.narrow(0, b, mini_batch_size))
        _,pred = output.max(1)
        
        for k in range(mini_batch_size):
            if target_[b+k] != pred[k]:
                nb_error += 1
                
    return nb_error