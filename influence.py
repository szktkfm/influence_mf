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
    y = torch.tensor([1.], device=device)
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


def hessian_vector_product(model, grad_loss, v):
    w = torch.dot(grad_loss, v)
    grads = torch.autograd.grad(w, model.parameters(), retain_graph=True)
    grad = []
    for g in grads:
        tmp = g.view(1, g.shape[0] * g.shape[1])
        grad.append(tmp)

    return torch.cat(grad, dim=1)[0]


def conjugate_grad(model, grad_loss, m, grad):
    a = torch.dot(m, hessian_vector_product(model, grad_loss, m))
    b = torch.dot(m, hessian_vector_product(model, grad_loss, grad))
    m = grad - (b / a) * m 

    return m

def hessian_inverse_vector_product(model, grad_loss, v):
    tol = 1e-7
    max_iter = 100
    # init
    x = torch.rand(v.size(), device=device)
    grad = hessian_vector_product(model, grad_loss, x) - v
    m = grad
    lr = -1 * torch.dot(m, grad) / torch.dot(m, hessian_vector_product(model, grad_loss, m))
    x = x + lr * m

    # 共役勾配法で解く
    for i in range(max_iter):
        pre_x = x
        grad = hessian_vector_product(model, grad_loss, x) - v
        m = conjugate_grad(model, grad_loss, m, grad)
        lr = -1 * torch.dot(m, grad) / torch.dot(m, hessian_vector_product(model, grad_loss, m))
        x = x + lr * m
        if torch.norm(pre_x - x) < tol * x.shape[0]:
            break

    return x


def get_influence(loss_func, train_data, test_data, model):
    loss = get_loss(loss_func, train_data, model)

    grad_loss = get_loss_gradient(model, loss)
    grad_pred = predict_gradient(test_data, model)
    h_inverse_v = hessian_inverse_vector_product(model, grad_loss, grad_pred)
    influ = -1 * torch.dot(h_inverse_v, grad_loss) / grad_loss.shape[0] 

    return influ