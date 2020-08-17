import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MF(nn.Module):


    def __init__(self, embedding_dim, user_size, item_size, layer_size):
        super(MF, self).__init__()
        self.embedding_dim = embedding_dim

        self.user_embed = nn.Embedding(user_size, embedding_dim)
        self.item_embed = nn.Embedding(item_size, embedding_dim)

        self.user_lin = nn.Embedding(user_size, 1)
        self.item_lin = nn.Embedding(item_size, 1)

        
    def forward(self, user_tensor, item_tensor):
        # user, itemをembed
        user_embed = self.user_embed(user_tensor)
        item_embed = self.item_embed(item_tensor)
        
        interaction_embed = user_embed * item_embed
        batch_size = interaction_embed.shape[0]

        bias = self.user_lin(user_tensor) + self.item_lin(item_tensor)
        prob = torch.sigmoid(torch.sum(interaction_embed, 1) + bias.view(batch_size))
        
        return prob
    
    def predict(self, user_tensor, item_tensor):
        # user, itemをembed
        user_embed = self.user_embed(user_tensor)
        item_embed = self.item_embed(item_tensor)
        
        interaction_embed = user_embed * item_embed
        batch_size = interaction_embed.shape[0]

        bias = self.user_lin(user_tensor) + self.item_lin(item_tensor)
        prob = torch.sigmoid(torch.sum(interaction_embed, 1) + bias.view(batch_size))
        
        return prob
