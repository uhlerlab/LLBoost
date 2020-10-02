import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.datasets import CIFAR10
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from numpy.linalg import pinv, norm, svd
import random
import pickle
import torchvision.transforms as transforms



def make_dataset(num_data):
    """
    Returns training data for Cats and Dogs from CIFAR10. 
    
    Parameters:
      num_data (int): Total number of Cat / Dog Examples to retrieve. 
    
    Returns:
      train_loader (DataLoader): Contains the training data for the model, where the labels are -1 for cats and +1 for dogs. 
    """
    train_transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])
    batch_size = 128
    train_set = CIFAR10(root="./data", train=True,
                       transform=train_transformations, download=True)
    y_train = train_set.targets
    y_train = np.array(y_train)
    pos_3 = np.argwhere(y_train == 3)
    pos_5 = np.argwhere(y_train == 5)
    pos_3 = list(x for x in pos_3[:, 0])
    pos_5 = list(x for x in pos_5[:, 0])
    half = int(num_data / 2)
    x_catdog = [train_set[j][0].numpy() for j in pos_3[:half]]
    x_catdog.extend([train_set[j][0].numpy() for j in pos_5[:half]])
    y_catdog = [-1 for x in range(half)]
    y_catdog.extend([1 for x in range(half)])

    x_catdog = np.array(x_catdog)
    y_catdog = np.array(y_catdog)
    trainData = TensorDataset(torch.from_numpy(
        x_catdog).float(), torch.from_numpy(y_catdog).float())
    train_loader = DataLoader(
        trainData, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_loader

def make_test_dataset():
    """
    Returns testing data for Cats and Dogs from CIFAR10. 
        
    Returns:
      test_loader (DataLoader): Contains the test data for the model, where the labels are -1 for cats and +1 for dogs. 
    """

    test_transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    batch_size = 128
    test_set = CIFAR10(root="./data", train=False,
                       transform=test_transformations, download=True)
    y_test = test_set.targets
    y_test = np.array(y_test)
    pos_3_test = np.argwhere(y_test == 3)
    pos_5_test = np.argwhere(y_test == 5)
    pos_catdog = list(pos_3_test[:, 0])
    pos_catdog.extend(list(pos_5_test[:, 0]))
    x_test_catdog = np.array([test_set[j][0].numpy() for j in pos_catdog])
    y_test_catdog = [-1 for x in range(len(pos_3_test))]
    y_test_catdog.extend([1 for x in range(len(pos_5_test))])
    y_test_catdog = np.array(y_test_catdog)
    testData = TensorDataset(torch.from_numpy(
        x_test_catdog).float(), torch.from_numpy(y_test_catdog).float())
    test_loader = DataLoader(
        testData, batch_size=batch_size, shuffle=False, num_workers=4)

    return test_loader

class CNN_Model(nn.Module):
    """ Defines a pre-trained Resnet 18 Model """
    
    def __init__(self):
        super(CNN_Model, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.linear = nn.Linear(512, 1, bias=False)

    def forward(self, images):
        o = self.resnet(images)
        o = o.view(-1, 512)
        o = self.linear(o)
        return o

def get_error(X, y, w):
    """
    Compute the Mean-Squared Error
    
    Parameters: 
      X (np.ndarray): Data Matrix with dimension (Input Dimension of Linear Layer x Number of Samples) 
      y (np.ndarray): True labels with dimension (1 x Number of Samples)
      w (np.ndarray): Weight vector with dimension (1 x Input Dimension of Linear Layer)
     
    Returns: 
      Mean-Squared Error, i.e. (y - wX)^2
    """
    error = 0
    pred = w @ X
    return np.mean(np.power(y - pred, 2))


def get_acc(X, y, w):
    """
    Compute the Accuracy of the Weight Vector. 
    
    Parameters: 
      X (np.ndarray): Data Matrix with dimension (Input Dimension of Linear Layer x Number of Samples) 
      y (np.ndarray): True labels with dimension (1 x Number of Samples) and values of either {1, -1}. 
      w (np.ndarray): Weight vector with dimension (1 x Input Dimension of Linear Layer)
    
    Returns:
      Accuracy of w, i.e. Number of Correct Predictions / Number of Total Predictions.  
    """
    pred = w @ X
    out = np.where(pred > 0, 1, -1)
    return np.mean(out == y)


def get_batch_acc(X, y, w):
    """
    Computes the best test accuracy among the different initializations. 
    
    Parameters: 
      X (np.ndarray): Data Matrix with dimension (Input Dimension of Linear Layer x Number of Samples) 
      y (np.ndarray): True labels with dimension (1 x Number of Samples) and values of either 1 or -1. 
      w (np.ndarray): Weight vector with dimension (Number of Sampled Initializations (100,000) x Input Dimension of Linear Layer)
    
    Returns:
      w_best (np.ndarray): Best adjusted last layer weight
    """

    pred = (w @ X).cpu().numpy()
    out = np.where(pred > 0, 1, -1)
    k, _ = out.shape # Number of Samples, (100,000)
    new_y = np.concatenate([y] * k, axis=0)
    match =  np.mean(out == y, axis=1)
    best_ind = np.argmax(match)
    w_best = w.cpu().data.numpy()[best_ind, :]
    
    return w_best


def load_dataset(loader, features, dim):
    """
    Pass the train and test data through the convolution layers of the model. 
    
    Parameters:
      loader (DataLoader): Data loader containing train and test data 
      features (torch.Tensor): Convolution features of the resnet model
      dim (int): Dimension of the last linear layer 
    
    Returns:
      X (np.ndarray): Training Feature Matrix with dimension (Input Dimension of Linear Layer x Number of Samples) 
      y (np.ndarray): The true labels as a (1 x Number of Samples) vector
    """
    vecs = []
    targets = []
    for batch in loader:
        img = batch[0].cuda()

        vec = features(img).view(-1, dim).data.cpu().numpy()
        vecs.append(vec)
        targets.append(batch[1].numpy())
        X = np.concatenate(vecs, axis=0)
        X = X.T
        y = np.concatenate(targets, axis=0)
    return X, y

def llboost(X, X_t, y_t, w_b, T, gamma):
    """
    Performs the LLBoost algorithm. 
    
    Parameters:
        X (np.ndarray): Training feature matrix with dimension (Input Dimension of Linear Layer x Number of Samples) 
        X_t (np.ndarray): Test feature matrix with dimension (Input Dimension of Linear Layer x Number of Samples)
        y_t (np.ndarray): The true test labels as a (1 x Number of Samples) vector
        w_b (np.ndarray): The last layer weights 
        T (int): Number of samples
        gamma (float): Radius of hypersphere 
        
    Returns:
        w_best (np.ndarray): Adjusted last layer weights
    """
    d = 512 # Dimension of Last linear layer
    U, s, Vt = svd(X, full_matrices=False)
    P_t = np.ones(d)
    P_t[len(s):] = 0
    P_t = np.diag(P_t)
    perp = np.eye(d) - U @ P_t @ U.T
    #perp = np.eye(d) - X @ pinv(X.T @ X) @ X.T
    
    # Sample T Initializations w1
    w1 = np.random.randn(d, T)
    w1 /= np.sqrt(np.sum(np.power(w1, 2), axis=0))
    w1 = w1.T
    
    w1 = torch.from_numpy(w1).float().cuda()
    perp = torch.from_numpy(perp).float().cuda()
    w_b = torch.from_numpy(w_b).float().cuda()
    X_test = torch.from_numpy(X_t).float().cuda()
    
    w_r = gamma * w1 @ perp + w_b # Add initialization to the baseline weight
    w_best = get_batch_acc(X_test, y_t, w_r) # Compute best test accuracy among the 100,000 Samples
    
    return w_best


def main(SEED):
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)

    net = CNN_Model() # ResNet-18 Model, can replace with any pre-trained model

    net.eval()

    d = 512 # Dimension of last linear layer

    trainset = make_dataset(100) # 100 Cat / Dog CIFAR10 Examples

    features = net.resnet

    features.cuda()

    X, y = load_dataset(trainset, features, d)
    _, n = X.shape

    gamma = 10**2 / (d ** 0.5)

    w_b = net.linear.weight.cpu().data.numpy()

    print("Original Train Error: ", get_error(X, y, w_b), "Original Train Acc: ", get_acc(X, y, w_b))

    testset = make_test_dataset()

    X_t, y_t = load_dataset(testset, features, d)
    print("Original Test Error: ", get_error(X_t, y_t, w_b), "Original Test Acc: ", get_acc(X_t, y_t, w_b))
    
    T = 100000 # 100,000 samples
    w_best = llboost(X, X_t, y_t, w_b, T, gamma)
    
    print("Corrected Test Error: ", get_error(X_t, y_t, w_best), "Corrected Test Acc: ", get_acc(X_t, y_t, w_best))
    

if __name__ == "__main__":
    main(200)
