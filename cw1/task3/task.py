#import libraries
# import tensorflow as tf
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset, random_split
from PIL import Image
import numpy as np
import time

from utils import cutout
from network_pt import Net
from model1 import LinearNet
from model2 import CNN

#define kfold function
def kfold(fold, trainset, net, opt, loss_fc, epochs=2, batch =20, use_cutout = False, name = None):

    '''
    Functiont that runs crossvalidation does not return obj., instead prints the followin results for training/validation/testing:
    loss, accuracy and time to train/validate/ and test

    Input:
    fold = number of folds
    trainset = training set
    net = model
    opt = optimizer for this particular model
    loss_fc = loss function/criterion
    epochs = how many epochs to run, default is 2
    batch = number of minibatches, default is 20
    use_cutout = binary flag, indicates whether training should use the cutout algorithm or not
    name = string input, name to save the best performing model under
    '''
    
    accuracy = []
    
    t0 = time.perf_counter()

    #3 fold cross validation:
    for k in range(fold):

        print("\nStarting Fold", k)

        #split training into training and validation
        train, valid = random_split(trainset, [35000,15000])
        
        print("Data Split!", "\nTraining Set:", len(train), "images", "\nValidation Set:", len(valid), "images")

        trainloader = torch.utils.data.DataLoader(train, batch_size=batch, shuffle=True, num_workers=2)
        validloader = torch.utils.data.DataLoader(valid, batch_size=batch, shuffle=True, num_workers=2)

        #initialize the net
        model = net

        #reset weights
        if k == 0:
            torch.save(model.state_dict(), 'reset_model.pt')
        else:
            model.load_state_dict(torch.load('reset_model.pt'))

        #define loss and optimizer
        criterion = loss_fc
        optimizer = opt

        print("\nStart Training")

        t_start = time.perf_counter()

        for epoch in range(epochs):
            
            print("Starting Epoch", epoch)

            running_loss = 0.0
            correct = 0.0
            total = 0

            for i,data in enumerate(trainloader,0):

                inputs, labels = data
                
                if use_cutout == True:
                    #insert cutout algorithm into training
                    cutouts = []
                    for j,x in enumerate(inputs):
                        new_image = cutout(1,16,x)
                        cutouts.append(new_image)
            
                    inputs = torch.stack(cutouts)


                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _,predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_loss = running_loss/len(trainloader)
            train_accu = 100.*correct/total
            print('Train Loss: %.3f | Train Accuracy: %.3f'%(train_loss,train_accu),'%')
        
        t_end = time.perf_counter()

        print('Training done. Elapsed Time:', t_end - t_start)

        # save trained model
        save_path = f'./{k}_fold_model.pt'
        torch.save(model.state_dict(),save_path)
        print('Model saved.')

        print("\nStarting Validation")

        running_loss = 0.0
        correct = 0.0
        total = 0
        t_start = time.perf_counter()

        with torch.no_grad():
            for i,data in enumerate(validloader, 0):
                inputs, labels = data

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _,predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        t_end = time.perf_counter()
        
        valid_loss = running_loss/len(validloader)
        valid_accu = 100.*correct/total
        accuracy.append(valid_accu)

        print('Validation Loss: %.3f | Validation Accuracy: %.3f'%(valid_loss,valid_accu),'%')
        print("Elapsed Time:", t_end-t_start)
    
    tl = time.perf_counter()
    print("Cross Validation Done!")
    print("Total Elapsed Time:", tl-t0)

    max_index = accuracy.index(max(accuracy))
    print("Fold", max_index, "is the best model")
    print("Accuracy is", max(accuracy), '%')
    
    load_path = f'./{max_index}_fold_model.pt'
    model.load_state_dict(torch.load(load_path))

    path = f'./{name}_Best_Model.pt'
    torch.save(model.state_dict(),path)

    print("\nBegin Testing with Best Model")

    running_loss = 0.0
    correct = 0.0
    total = 0
    t_start = time.perf_counter()

    with torch.no_grad():
        for i,data in enumerate(testloader, 0):
            inputs, labels = data

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _,predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    t_end = time.perf_counter()
        
    test_loss = running_loss/len(validloader)
    test_accu = 100.*correct/total

    print('Test Loss: %.3f | Test Accuracy: %.3f'%(test_loss,test_accu),'%')
    print("Elapsed Time:", t_end-t_start)



if __name__ == '__main__':
    ## cifar-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 20

    #get training set
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    #get test set
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    #declare classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print("Evaluate training with and without cutout data augmentation algorithm implemented in task2")

    #use model from tutorial
    net = Net()

    #define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=.001, momentum =.9)

    print("\nWith Cutout:")
    #use cross validation with cutout
    kfold(3, trainset, net, optimizer, loss_fc=criterion, epochs=2, batch=20, use_cutout=True, name = "Tutorial_with_Cutout")

    print("\nWithout Cutout:")
    #use cross validation without cutout
    kfold(3, trainset, net, optimizer, loss_fc=criterion, epochs=2, batch=20, use_cutout=False, name = "Tutorial")

    #Define new model
    net = LinearNet()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=.001, momentum =.9)

    print("\nPerform 3 fold cross validation on Model 1")

    #perform 3 fold cross validation on 1 model
    kfold(3, trainset, net, optimizer, loss_fc=criterion,epochs=2, batch = 20, use_cutout=False, name = "Model1")

    #Define new model
    net = CNN()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=.001, momentum =.9)

    print("\nPerform 3 fold cross validation on Model 2")

    #perform 3 orld cross validation on 2 model
    kfold(3, trainset, net, optimizer, loss_fc=criterion, epochs=2, batch = 20, use_cutout=False, name = "Model2")

