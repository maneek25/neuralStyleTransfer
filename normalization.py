
import torch
import torch.nn as nn

#Normalizes input image
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()

        #Copies tensors and detches tensor from previous graph
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, image):
        #Normalize image
        return (image - self.mean) / self.std