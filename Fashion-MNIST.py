import torch
from torchvision import datasets, transforms
import helper
import matplotlib
# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

import torch
from torch import nn, optim

# Network architecture here
Classifier = nn.Sequential(nn.Linear(784, 300),
                     nn.ReLU(),
                     nn.Linear(300, 150),
                     nn.ReLU(),
                     nn.Linear(150, 10),
                     nn.LogSoftmax(dim = 1))

# Define the loss
Criterion = nn.CrossEntropyLoss()
# Optimizer
optimizer = optim.SGD(Classifier.parameters(), lr=0.003)

# Train the network 
epochs = 25


for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        
        #Flatten image
        # ~ images.resize_(64, 784)
        images = images.view(images.shape[0], -1)
        optimizer.zero_grad()

        # Pass forward
        output = Classifier(images)

        # Calculate loss 
        # Note: for some reason, if I put label without s, it gives me an error of batch size 32 not matching 64
        loss = Criterion(output, labels)
        # Calculate gradient
        loss.backward()
        # Optimize
        optimizer.step()
        
        #Add loss
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")


import torch.nn.functional as F
import helper

# Test out your network!

dataiter = iter(testloader)
images, labels = next(dataiter)
img = images[0]
# Convert 2D image to 1D vector
img = img.resize_(1, 784)

# TODO: Calculate the class probabilities (softmax) for img
output = Classifier(img)
ps = F.softmax(output, dim = 1)

# Plot the image and probabilities
helper.view_classify(img.resize_(1, 28, 28), ps, version='Fashion')
