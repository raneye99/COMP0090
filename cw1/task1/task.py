#import from libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

#import own functions
from task1_func import polynomial_fun, fit_polynomial_ls, fit_polynomial_sgd

#Create training and test set
w = torch.tensor([1,2,3,4])

x_train = torch.distributions.uniform.Uniform(-20,20).sample([100,])
x_test =  torch.distributions.uniform.Uniform(-20,20).sample([50,])

y_train = polynomial_fun(w,x_train) + torch.normal(0,.2, size = x_train.shape)
y_test = polynomial_fun(w,x_test) + torch.normal(0,.2,size = x_test.shape)

#Use fit_polynomial_ls(M=4) to compute the optimum weight vector w using the trianing set
w_hat_ls = fit_polynomial_ls(x_train,y_train,4)

#Then predict target values for trianing and test set
y_train_hat_ls = polynomial_fun(w_hat_ls, x_train)
y_test_hat_ls = polynomial_fun(w_hat_ls, x_test)

# Report using printed messages the mean and std in difference betwen 
# A) the observed training data and the underlying true polynomial curve
print("A:")
diff_obs = y_train - polynomial_fun(w, x_train)
mean_obs = torch.mean(diff_obs)
std_obs = torch.std(diff_obs)

print("The mean in difference between the observed training data and the underlying true polynomial curve is: %5.4f" % mean_obs.data.tolist())
print("The standard deviation in difference between the observed training data and the underlying true polynomial curve is: %5.2f" % std_obs.data.tolist())

# B) the predicted values and the true polynomial curve
print("B:")
diff_pred_train_ls = y_train_hat_ls - polynomial_fun(w, x_train)
mean_pred_train_ls = torch.mean(diff_pred_train_ls)
std_pred_train_ls = torch.std(diff_pred_train_ls)
print("The mean in difference between the predicted training data for least squares and the underlying true polynomial curve is: %5.4f" % mean_pred_train_ls.data.tolist())
print("The standard deviation in difference between the predicted training data for least squares and the underlying true polynomial curve is: %5.2f" % std_pred_train_ls.data.tolist())

diff_pred_test_ls = y_test_hat_ls - polynomial_fun(w, x_test)
mean_pred_test_ls = torch.mean(diff_pred_test_ls)
std_pred_test_ls = torch.std(diff_pred_test_ls)
print("The mean in difference between the predicted test data and the underlying true polynomial curve is: %5.4f" % mean_pred_test_ls.data.tolist())
print("The standard deviation in difference between the predicted test data and the underlying true polynomial curve is: %5.2f" % std_pred_test_ls.data.tolist())

#Use fit_polynomial_sgd(M=4) to optimize the wight vector using the trianing set
w_hat_sgd = fit_polynomial_sgd(x_train,y_train,4,1e-5,5)

#Then compute the predicted values for training and test set
y_train_hat_sgd = polynomial_fun(w_hat_sgd, x_train)
y_test_hat_sgd = polynomial_fun(w_hat_sgd, x_test)

# Report using printed messages the difference between predicted values and the underlying polynomial curve
diff_pred_train_sgd = y_train_hat_sgd - polynomial_fun(w, x_train)
mean_pred_train_sgd = torch.mean(diff_pred_train_sgd)
std_pred_train_sgd = torch.std(diff_pred_train_sgd)
print("The mean in difference between the predicted training data for least squares and the underlying true polynomial curve is: %5.4f" % mean_pred_train_sgd.data.tolist())
print("The standard deviation in difference between the predicted training data for least squares and the underlying true polynomial curve is: %5.2f" % std_pred_train_sgd.data.tolist())

diff_pred_test_sgd = y_test_hat_sgd - polynomial_fun(w, x_test)
mean_pred_test_sgd = torch.mean(diff_pred_test_sgd)
std_pred_test_sgd = torch.std(diff_pred_test_sgd)
print("The mean in difference between the predicted test data and the underlying true polynomial curve is: %5.4f" % mean_pred_test_sgd.data.tolist())
print("The standard deviation in difference between the predicted test data and the underlying true polynomial curve is: %5.2f" % std_pred_test_sgd.data.tolist())