#train script
# adapted from image classification tutorial: https://github.com/YipengHu/COMP0090/blob/main/tutorials/img_cls/train_pt.py

#import libraries
# import tensorflow as tf
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

#import our densenet
import network_pt as model
#import cutout algorithm
from utils import cutout


if __name__ == '__main__':
    ## cifar-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 20

    #get training set
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    #get test set
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    #declare classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # # example images
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()

    # im = Image.fromarray((torch.cat(images.split(1,0),3).squeeze()/2*255+.5*255).permute(1,2,0).numpy().astype('uint8'))
    # im.save("train_pt_images.jpg")
    # print('train_pt_images.jpg saved.')
    # print('Ground truth labels:' + ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    ## densenet
    net = model.DenseNet3()
    ##print model architexture
    print("Model Architecture:")
    print("\n",net)


    ## loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    #initialize empty vectors to store values
    train_accuracy = []
    train_losses =  []
    test_accuracy = []
    test_losses = []

    ## train
    for epoch in range(10):  # loop over the dataset multiple times

        print('\nEpoch: %d'%epoch+1)

        #train
        running_loss = 0.0
        correct = 0.0
        total = 0
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            #insert cutout algorithm into training
            cutouts = []
            for j,x in enumerate(inputs):
                new_image = cutout(1,16,x)
                cutouts.append(new_image)
            
            new_inputs = torch.stack(cutouts)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(new_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            
            _,predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss/len(trainloader)
        accuracy = 100.*correct/total

        train_accuracy.append(accuracy)
        train_losses.append(train_loss)
        print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss,accuracy))

        #test
        running_loss = 0.0
        correct = 0.0
        total = 0

        #forward pass only
        with torch.no_grad():
            for i, data in enumerate(trainloader,0):
                    

    print('Training done.')

    # save trained model
    torch.save(net.state_dict(), 'saved_model.pt')
    print('Model saved.')