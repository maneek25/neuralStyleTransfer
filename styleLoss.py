
import torch
import torch.nn.functional as F
import torch.nn as nn

#Calculates gram matrix in order to find feature correlations
def gram_matrix(input):
    a, b, c, d = input.size()  
    features = input.view(a * b, c * d) 

    #Computes the gram product
    gProd = torch.mm(features, features.t())  

    #Normalizes values by dividing by number of elements in each feature maps.
    return gProd.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        #Calculates gram matrix and detaches tensor from the graph
        self.target = gram_matrix(target_feature).detach()

    #Style Loss calculated during forward pass
    def forward(self, input):
        G = gram_matrix(input)
        #Calculates the mean squared error which is the Style Loss
        self.loss = F.mse_loss(G, self.target)
        return input
