#import libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image, ImageDraw, ImageFont
import numpy as np

#import our densenet
import network_pt as dn
#import cutout algorithm
from utils import cutout


if __name__ == '__main__':
    ## cifar-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 40

    #get training set
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    #get test set
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    #declare classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    ## densenet
    net = dn.DenseNet3()
    ##print model architexture
    print("Model Architecture:")
    print("\n",net)

    #define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=.001, momentum =.9)

    #initialize empty vectors to store values
    train_accuracy = []
    train_losses =  []
    test_accuracy = []
    test_losses = []

    ## train
    for epoch in range(10):  # loop over the dataset multiple times

        print(f'Starting Epoch {epoch+1}')

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
                new_image = cutout(1,32,x)
                cutouts.append(new_image)
            
            inputs = torch.stack(cutouts)

            #save cutout image for first batch
            if(epoch == 0):
                if (i == 0):

                    imgs = []
                    # get images
                    for k in range(len(inputs)):

                        img = (inputs[k]/2 +.5)*100  # unnormalize
                        im = img.cpu().detach().numpy()
                        # im = img.numpy()
                        im = np.transpose(im, (1,2,0))
                        im = np.uint8(im)
                        imgs.append(Image.fromarray(im, 'RGB'))

                    #create collage
                    collage = Image.new('RGB', (32, 32*16))
                    for l in range(16):
                        collage.paste(imgs[l], (0,32*l))
                        collage.save('cutout.png')

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            
            _,predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        
        train_loss = running_loss/len(trainloader)
        train_accu = 100.*correct/total

        print('Train Loss: %.3f | Train Accuracy: %.3f'%(train_loss,train_accu),'%')

        #test
        running_loss = 0.0
        correct = 0.0
        total = 0

        #forward pass only
        with torch.no_grad():
            for i, data in enumerate(testloader,0):
                inputs, lables = data

                outputs = net(inputs)

                loss = criterion(outputs,labels)
                running_loss+=loss.item()

                _,predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                #save a png of results 
                if (epoch == 9):
                    if (i == 0):
                        imgs = []
                        # get images
                        for k in range(len(inputs)):

                            img = (inputs[k]/2 +.5)*100  # unnormalize
                            im = img.cpu().detach().numpy()
                            im = np.transpose(im, (1,2,0))
                            im = np.uint8(im)
                            im = Image.fromarray(im, 'RGB')
                            im = im.resize((128,128))
                            im_draw = ImageDraw.Draw(im)
    
                            gt = "\nGT:"
                            pd = "Pred:"
                            tv = classes[labels[k]]
                            yh = classes[predicted[k]]
                            text = pd + yh + gt + tv
                            fontsize = 1
                            im_draw.text((0,0),text)

                            imgs.append(im)

                        #create collage
                        collage = Image.new('RGB', (128, 128*36))
                        for l in range(36):
                            collage.paste(imgs[l], (0,128*l))
                            collage.save('results.png')

        test_loss = running_loss/len(testloader)
        test_accu = 100.*correct/total

        test_losses.append(test_loss)
        test_accuracy.append(test_accu)

        print('Test Loss: %.3f | Test Accuracy: %.3f'%(test_loss,test_accu),'%')
                    

    print('Training and testing done.')

    # save trained model
    torch.save(net.state_dict(), 'saved_model.pt')
    print('Model saved.')