import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

class NN(nn.Module):

    # nObservations, nActions were originally inputSize and hiddenSize

    def __init__(self, nObservations = 1, nActions = 64):
        super(NN, self).__init__()

        # I forgot the impact inputSize and hiddenSize has. Just play around I guess

        self.L1 = nn.Linear(nObservations, nActions)
        self.L2 = nn.Linear(nActions, nActions)
        self.L3 = nn.Linear(nActions, nActions)

    def forward(self, x):

        # The activation functions are nonlinear for depth learning
        # Use Sigmoid if that is better

        # Forward pass
        x = F.relu(self.L1(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.L2(x))
        x = F.dropout(x, p=0.5)
        x = self.L3(x)

        return x
    
class NNfilter:

    def __init__(self):
        self.model = NN(nObservations  = 1, nActions = 64)

        # Using ADAM because we want to adjust the learning rate for reinforcement learning
        # ADAM works on CPU, CUDA, MPS
        # May want to convert to SGD
        # THIS PORTION NEEDS RESEARCH

        '''
        # Pre-parameter options for per-layer learning rates
        # INSERT CODE FOR ABOVE 

        # SGD optimizer
        optimizer = optim.SGD(model.parameters(), lr =0.01, momentum-0.9)



        '''

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        # Adjust number of learning rate based on the number of epochs

    def training(self, currentState, feedback, adjustLearningRate, inputWeights, outputWeight, desiredPoints):

        # Set the range here
        RANGE1 = -1
        RANGE2 = 1

        self.model.train() # Initzalize 

        tensorsInput = torch.tensor(currentState, dtype=torch.float32).view(RANGE1, RANGE2)
        tensorsInput = tensorsInput * torch.tensor(inputWeights, dtype=torch.float32).view(RANGE1, RANGE2)

        self.optimizer.zero_grad()

        tensorsOutput = self.model(tensorsInput)

        target = torch.tensor([desiredPoints], dtype=torch.float32).view(RANGE1, RANGE2)
        
        loss = nn.functional.mse_loss(tensorsOutput, target)


        # MSE loss computation, may want to check the math here
        # negative values should act as penalty

        if feedback > 0:
            loss *= (1.0 - feedback * 0.1)
        elif feedback < 0:
            loss *= (1.0 + abs(feedback) * 0.1)

        loss.backward()
        
        self.optimizer.step()

        adjustLearningRate *= 1 - 0.1 * abs(feedback)

        # param_group for fine tuning pre-trained network
        
        for param_groups in self.optimizer.param_groups:
            param_groups['lr'] = adjustLearningRate

    def predict(self, inputs, appendWeights):

        # Set the range here, make it global later I'm lazy
        RANGE1 = -1
        RANGE2 = 1

        with torch.no_grad():

            self.model.eval()

            tensorsInput = torch.tensor(inputs, dtype=torch.float32).view(RANGE1, RANGE2)
            tensorsInput = tensorsInput * torch.tensor(appendWeights, dtype=torch.float32).view(RANGE1, RANGE2)

            tensorsOutput = self.model(tensorsInput)
            return tensorsOutput.view(-1).tolist()
    
    def loadModel(self, fname):
        self.model.load_state_dict(checkpoint['optimizer_state_dict'])
        self.optimizer.load_state_dict([checkpoint['optimizer_state_dict']])
    
    def saveModel(self, fname):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, fname)






        

