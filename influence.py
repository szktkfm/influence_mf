from model import MF
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_loss(loss_func, train_data, model):
    user = torch.tensor([train_data[0]], dtype=torch.long, device=device)
    item = torch.tensor([train_data[1]], dtype=torch.long, device=device)
    pred = model(user, item)
    y = torch.tensor([1.])
    loss = loss_func(pred, y)

    return loss


def get_loss_gradient(model, loss):
    grads = torch.autograd.grad(loss, model.parameters(), create_graph = True, retain_graph = True)
    grad = []
    for g in grads:
        tmp = g.view(1, g.shape[0] * g.shape[1])
        grad.append(tmp)

    return torch.cat(grad, dim=1)[0]


def predict_gradient(test_data, model):
    model.zero_grad()
    user = torch.tensor([test_data[0]], dtype=torch.long, device=device)
    item = torch.tensor([test_data[1]], dtype=torch.long, device=device)
    pred = model(user, item)

    grads = torch.autograd.grad(pred, model.parameters(), create_graph = True, retain_graph = True)
    grad = []
    for g in grads:
        tmp = g.view(1, g.shape[0] * g.shape[1])
        grad.append(tmp)

    return torch.cat(grad, dim=1)[0]


# うまくやりたい
def hessian_vector_product(model, grad_loss, v):
    w = torch.dot(grad_loss, v)
    w.backward()
    params = model.parameters()
    hv = []
    for p in params:
        hv.append(p.grad.view(1, p.shape[0]*p.shape[1]))

    return torch.cat(hv, dim=1)[0]



def hessian_inverse_vector_product():
    return 0


def get_influence(loss_func, train_data, test_data, model):
    loss = get_loss(loss_func, train_data, model)

    grad_loss = get_loss_gradient(model, loss)
    grad_pred = predict_gradient(test_data, model)
    hv = hessian_vector_product(model, grad_loss, grad_pred)

    return hv