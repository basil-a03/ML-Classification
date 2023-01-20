import torch
import matplotlib.pyplot as plt
import helper
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

## TODO: Define your model with dropout added

class Classifier(nn.Module):
    def __init__(self):
        
        super().__init__()
        self.fc1 = nn.Linear(784, 377)
        self.fc2 = nn.Linear(377, 153)
        self.fc3 = nn.Linear(153, 75)
        self.fc4 = nn.Linear(75, 10)
        
        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim = 1)
        return x

## TODO: Train your model with dropout, and monitor the training progress with the validation loss and accuracy

model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 30
steps = 0

train_losses, test_losses = [], []
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        
        optimizer.zero_grad()
        
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        
    else:
        tot_test_loss = 0
        test_correct = 0
        
        # Validation pass and print out the validation accuracy
        # turn off gradients
        with torch.no_grad():
            
            # set model to evaluation mode
            model.eval()
            
            # validation pass here
            for images, labels in testloader:
                # Forward
                log_ps = model(images)
                
                # Estimate the number of loss
                loss = criterion(log_ps, labels)
                tot_test_loss += loss.item()
                
                # Estimate the number of correct
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                test_correct += equals.sum().item()
        
        
        
# Get mean loss to enable comparison between train and test sets
train_loss = running_loss / len(trainloader.dataset)
test_loss = tot_test_loss / len(testloader.dataset)
# At completion of epoch
train_losses.append(train_loss)
test_losses.append(test_loss)

# set model back to train mode
model.train()

print("Epoch: {}/{}.. ".format(e+1, epochs),
        "Training Loss: {:.3f}.. ".format(train_loss),
        "Test Loss: {:.3f}.. ".format(test_loss),
        "Test Accuracy: {:.3f}".format(test_correct / len(testloader.dataset)))   # testloader.dataset = batch size             


model.eval()

dataiter = iter(testloader)
images, labels = next(dataiter)
img = images[0]
# Convert 2D image to 1D vector
img = img.view(1, 784)

# Calculate the class probabilities (softmax) for img
with torch.no_grad():
    output = model.forward(img)

ps = torch.exp(output)

# Plot the image and probabilities
helper.view_classify(img.view(1, 28, 28), ps, version='Fashion')