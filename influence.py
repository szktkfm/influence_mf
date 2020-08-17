from model import MF
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_loss_gradient(loss_func, train_data, model):
    user = torch.tensor([train_data[0]], dtype=torch.long, device=device)
    item = torch.tensor([train_data[1]], dtype=torch.long, device=device)
    pred = model(user, item)
    y = torch.tensor([1.])
    loss = loss_func(pred, y)
    loss.backward()

    params = model.parameters()
    grad = []
    for p in params:
        tmp = p.grad.view(1, p.shape[0] * p.shape[1])
        grad.append(tmp)

    return torch.cat(grad, dim=1)


def predict_gradient(model, train_data):
    return 0

def hessian_vector_product():
    return 0

def hessian_inverse_vector_product():
    return 0

def get_influence(loss_func, train_data, model):
    loss_grad = get_loss_gradient(loss_func, train_data, model)

    return loss_grad