from model import MF
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



def loss_gradient(loss_func, pred, y):
    return 0

def predict_gradient(model, train_data):
    return 0

def hessian_vector_product():
    return 0

def hessian_inverse_vector_product():
    return 0

def influence(loss_func, train_data, model):
    return 0