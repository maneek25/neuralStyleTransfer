
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import copy

from normalization import Normalization
from styleLoss import StyleLoss
from contentLoss import ContentLoss
from PIL import Image

epochs = 500

#Processes an images in order to prepare it for neural style transfer
def loadImage(imageName):
    image = Image.open(imageName)           #Loads image from a file      
    image = image.resize((600, 800))        #Resizes the image
    loader = transforms.Compose([               
        transforms.Resize(imsize),          #Scales image based on whether its CUDA or CPU
        transforms.ToTensor()])             #Transforms the image into a torch tensor
    image = loader(image).unsqueeze(0)      #Adds a dimension of size 1 to the torch tensor
    return image.to(device, torch.float)

#Reconverts torch tensor back to image and displays it
def showImg(tensor, title, runs):
    image = tensor.cpu().clone()            #Clones the tensor so that it doesn't make changes on it
    image = image.squeeze(0)                #Removes the additional batch dimension
    unloader = transforms.ToPILImage()      #Reconverts into a PIL image
    image = unloader(image)                 
    plt.title(title)
    plt.axis('off')
    if runs < epochs:                        #Runs is not the final run so only displays image
        plt.imshow(image, interpolation='nearest')
        plt.savefig('screengrabs/{}.jpg'.format(runs), bbox_inches="tight")
        plt.pause(1)
    else:                                   #Runs is final run so saves image to outputs and displays it     
        plt.imshow(image, interpolation='nearest')
        plt.savefig('screengrabs/{}.jpg'.format(runs), bbox_inches="tight")
        plt.savefig('output/output.jpg', bbox_inches="tight")
        plt.pause(1)


#Performs neural style 
def runStyleTransfer(contentImg, styleImg, epochs, styleWeight=1000000, contentWeight=1):

    #Defines pretrained VGG19 network from torchvision
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    #Copy of content image. This image is the one that is style transferred
    inputImg = content.clone()

    # Run the style transfer
    print('Building the model...')

    cnn = copy.deepcopy(cnn)

    #Prevents overflow by normalizing data
    normalizationMean = torch.tensor([0.485, 0.456, 0.406]).to(device)  
    normalizationStd = torch.tensor([0.229, 0.224, 0.225]).to(device)
    normalization = Normalization(normalizationMean, normalizationStd).to(device)

    #The depth layers that will be used to compute style/content losses
    contentLayers = ['conv_4']
    styleLayers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    #Stores content and style losses in array in order to allow iteration
    contentLosses = []
    styleLosses = []

    #Torch tensor model used to add models sequentially
    model = nn.Sequential(normalization)

    i = 0  #Increments everytime there is a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):        #If it is a 2D Convolutional
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):        #If it is a Rectified Linear Unit
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):   #If it is a 2D Max Pooling
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d): #If it is Batch Normalization
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        #Adds layer and its name to the model
        model.add_module(name, layer)

        if name in contentLayers:
            #Add content loss
            target = model(contentImg).detach()
            contentLoss = ContentLoss(target)
            model.add_module("contentLoss_{}".format(i), contentLoss)
            contentLosses.append(contentLoss)

        if name in styleLayers:
            #Add style loss
            target_feature = model(styleImg).detach()
            styleLoss = StyleLoss(target_feature)
            model.add_module("styleLoss_{}".format(i), styleLoss)
            styleLosses.append(styleLoss)

    #Trims off layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    
    model = model[:(i + 1)]

    #Defines LBFGS optimizer algorithm that we use
    optimizer = optim.LBFGS([inputImg.requires_grad_()])

    run = [0]
    plt.figure()

    #Runs epochs number of iterations for style transfer
    while run[0] <= epochs:

        #Forward Pass
        def closure():
            #Corrects the values of updated input image
            inputImg.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(inputImg)

            #Calculates Loss
            styleScore = 0
            contentScore = 0
            for sl in styleLosses:
                styleScore += sl.loss
            for cl in contentLosses:
                contentScore += cl.loss
            styleScore *= styleWeight
            contentScore *= contentWeight
            loss = styleScore + contentScore

            #Backward Pass, Calculates Gradients
            loss.backward()             
            run[0] += 1
            runs = run[0]

            #Prints Style and Content loss and displays image every 50 iterations
            if run[0] % 50 == 0:
                print('Total runs: {} Style Loss : {:4f} Content Loss: {:4f}'.format(runs,
                                                                                     styleScore.item(),
                                                                                     contentScore.item()))
                showImg(inputImg, "Building Output: {} of {} runs".format(runs, epochs), runs)
                print()

            return styleScore + contentScore

        #Updates weights and optimizer
        optimizer.step(closure)

    #Corrects values a final time
    inputImg.data.clamp_(0, 1)

    return inputImg

# Image output will be bad and compromised if there is no cuda support.
# Use small image size if no gpu. Will take a lot of time without GPU with 512.
if (torch.cuda.is_available()):
    device = torch.device("cuda")
    imsize = 512
    print("Cuda Detected. The output quality will not be compromised")
else:
    device = torch.device("cpu")
    imsize = 128
    print("Using CPU. Output quality compromised to save time")    

#Loads Content and Style Images and displays them
style = loadImage("./images/style/scream.jpg")
content = loadImage("./images/content/7eleven.jpg")
plt.ion()
plt.figure()
showImg(style, title='Style Image', runs=0)
plt.figure()
showImg(content, title='Content Image', runs=0)

assert style.size() == content.size(), \
    "Style and Content images must have the same size"


runStyleTransfer(content, style, epochs)
