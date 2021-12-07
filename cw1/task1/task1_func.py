import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

def polynomial_fun(w,x):
    '''
    Inputs:
    w = tensor of weights (vector)
    x = tesnor of scalar values of x, our r.v. (can be vector)

    Outputs:
    y = tensor of the sum of our polynomial fit (if x is vector, also vector)
    '''
    
    m = len(w)
    n = len(x)

    basis = Variable(torch.zeros([n,m]))

    for i in range(m):
        basis[:,i] = torch.pow(x,i)

    y = torch.sum(w*basis,1, dtype=float)   

    return y

def fit_polynomial_ls(X,t,M):
    '''
    Inputs:
    t = tensor of target value (vector of size n)
    X = tensor of scalar values of x, our r.v. (vector of size n)
    M = how many powers in our polynomial fit (scalar)

    Outputs:
    w = tensor of weight values (vector of size M)
    '''

    #initialize empty vector to store feature map of basis functions
    n = len(X)
    basis = Variable(torch.zeros([n,M]))

    for i in range(M):
        basis[:,i] = torch.pow(X,i)

    #solve for weights: w = (X.T * X)^-1 *X.T*y
    A = torch.matmul(basis.T, basis)
    b = torch.matmul(basis.T, t)

    w = torch.matmul(torch.linalg.torch.pinverse(A),b)

    return w

def fit_polynomial_sgd(X, t, M, lr, b):
    '''
    Inputs:
    t = tensor of target value (vector of size n)
    X = tensor of scalar values of x, our r.v. (vector of size n)
    M = how many powers in our polynomial fit (scalar)
    lr = learning rate (scalar)
    b = batch size (scalar)

    Outputs:
    w = tensor of weight values (vector of size M)
    '''

    #combine dataset
    train_data = TensorDataset(X,t)

    #create batch loader
    train_loader = DataLoader(dataset = train_data, batch_size=b)

    #initialize weight
    w_hat = Variable(torch.ones(M, dtype = float), requires_grad = True)

    #Define loss func
    loss_fn = nn.SmoothL1Loss(reduction = 'sum')
    # losses = []

    # define an optimizer to update param
    optimizer = optim.SGD([w_hat], lr=lr , weight_decay = 1)

    for epoch in range(1000):
        for x_train, y_train in train_loader:

            y_hat = polynomial_fun(w_hat,x_train)

            loss = loss_fn(y_train, y_hat)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # losses.append(loss)

    return w_hat

