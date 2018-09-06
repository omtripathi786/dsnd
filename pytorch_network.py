import numpy as np
import torch
import helper
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
from collections import OrderedDict


def get_train_test():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])
    # Download and load the training data
    trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Download and load the test data
    testset = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    return trainloader, testloader


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x


def feed_forward_network():
    # Build a feed-forward network
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[1], output_size),
                          nn.Softmax(dim=1))
    return model


if __name__ == '__main__':
    train_loader, test_loader = get_train_test()
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    # plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')
    # plt.show()
    model1 = Network()
    """
    Model1
    Forward pass
    Now that we have a network, let's see what happens when we pass in an image. This is called the forward pass.
    We're going to convert the image data into a tensor, then pass it through the operations defined
    by the network architecture.
    """
    # Resize images into a 1D vector, new shape is (batch size, color channels, image pixels)
    images.resize_(64, 1, 784)
    # Forward pass through the network
    img_idx = 0
    ps = model1.forward(images[img_idx, :])
    img = images[img_idx]
    helper.view_classify(img.view(1, 28, 28), ps)

    """
    Model2
    PyTorch provides a convenient way to build networks like this where a tensor is passed sequentially through
    operations, nn.Sequential. Using this to build the equivalent network:
    """
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10
    model2 = feed_forward_network()
    ps = model2.forward(images[0, :])
    helper.view_classify(images[0].view(1, 28, 28), ps)

    """
    Model3
    feed forward network through nn.Sequential and orderdict
    """
    model3 = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_sizes[0])),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
        ('relu2', nn.ReLU()),
        ('logits', nn.Linear(hidden_sizes[1], output_size))]))

    """
    Training the network!
    The first thing we need to do for training is define our loss function. In PyTorch, you'll usually see this as
    criterion. Here we're using softmax output, so we want to use criterion = nn.CrossEntropyLoss() as our loss. 
    Later when training, you use loss = criterion(output, targets) to calculate the actual loss.
     
    We also need to define the optimizer we're using, SGD or Adam, or something along those lines.
    Here I'll just use SGD with torch.optim.SGD, passing in the network parameters and the learning rate.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model3.parameters(), lr=0.01)
    epochs = 3
    print_every = 40
    steps = 0
    for e in range(epochs):
        running_loss = 0
        for images, labels in iter(train_loader):
            steps += 1
            # Flatten MNIST images into a 784 long vector
            images.resize_(images.size()[0], 784)

            optimizer.zero_grad()

            # Forward and backward passes
            output = model3.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e + 1, epochs),
                      "Loss: {:.4f}".format(running_loss / print_every))

                running_loss = 0

    # Turn off gradients to speed up this part
    with torch.no_grad():
        logits = model3.forward(img)

    # Output of the network are logits, need to take softmax for probabilities
    ps = F.softmax(logits, dim=1)
    helper.view_classify(img.view(1, 28, 28), ps)