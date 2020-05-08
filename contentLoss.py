
import torch
import torch.nn.functional as F
import torch.nn as nn

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        #Detaches tensor from the graph
        self.target = target.detach()

    #Content Loss calculated during forward pass
    def forward(self, input):
        #Calculates mean squared error which is the Content Loss
        self.loss = F.mse_loss(input, self.target)
        return input