from task2vec import Task2Vec
from models import get_model
import datasets
import task_similarity
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as tt
import torch.nn as nn
from tqdm import tqdm
from synbols_utils import Synbols
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime


class HardAttentionRaw(nn.Module):
    """Attention through Raw Params
            Input: Param shape, No of Params"""
    def __init__(self, shape):
        super(AttentionRaw, self).__init__()
        # Attention Params: Shape N_attn, M as N Attentions of dimension M 
        self.params = nn.Parameter(torch.randn(N, task2vec_dim), requires_grad = True)

    def forward(self, x):
        # Weighted/Atttention on x using Attention 'i'
        # X shape: N_z, M, Params shape: N_attn, M
        # Do Element wise multiplication
        return torch.mul(torch.unsqueeze(self.params, 1), x) 

class AttentionRaw(nn.Module):
    """Attention through Raw Params
            Input: Param shape, No of Params"""
    def __init__(self, shape):
        super(AttentionRaw, self).__init__()
        # Attention Params: Shape N_attn, M as N Attentions of dimension M 
        self.params = nn.Parameter(torch.randn(N, task2vec_dim), requires_grad = True)

    def forward(self, x):
        # Weighted/Atttention on x using Attention 'i'
        # X shape: N_z, M, Params shape: N_attn, M
        # Do Element wise multiplication
        soft = F.softmax(self.params, dim = -1)
        return torch.mul(torch.unsqueeze(soft, 1), x) 

def cos_sim(A,B, eps=1e-8, ):
    A_norm = A.norm(dim = -1, keepdim = True)
    B_norm = B.norm(dim = -1, keepdim = True)
    A_norm = torch.maximum(A_norm, eps * torch.ones(A_norm.shape))
    B_norm = torch.maximum(B_norm, eps * torch.ones(B_norm.shape))
    A = torch.div(A,A_norm)
    B = torch.div(B,B_norm)
    return torch.mm(A,B.T)
def norm_unitvectors(M, eps = 1e-8):
    M_norm = M.norm(dim = -1, keepdim = True)
    M_norm = torch.maximum(M_norm, eps * torch.ones(M_norm.shape))
    return torch.div(M, M_norm)
def positives(Attrs):
    M = norm_unitvectors(Attrs)
    M_trans = M.transpose(1,2)
    return torch.bmm(M, M_trans)
def positive_loss(Attrs, log = True):
    if log:
        x = -torch.log(positives(Attrs).flatten())
    else:
        x = 1 - positives(Attrs).flatten().to(torch.float16)
    return x.sum()
def negative_loss(Attrs):
    """For two attributes
    """
    neg_pair = cos_sim(Attrs[0], Attrs[1])
    return torch.abs(neg_pair.flatten()).sum()
def train(attns, Z, optimizer, gamma = 2):
    attns.train()
    attrs = attns(Z)
    pos_loss = positive_loss(attrs)
    neg_loss = negative_loss(attrs)
    loss =  pos_loss + neg_loss * gamma
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return pos_loss, neg_loss, loss


if __name__ == '__main__':
    saved = torch.load('all.pt')
    seed = 123
    ref = pd.read_csv('ref.csv', index_col = 0)
    Z = [z.hessian for z in saved]
    task2vec_dim = Z[0].shape[0]
    latent_dim = 128
    M = len(Z) # No of Task vectors
    Z_tensor = torch.tensor(Z)

    N = 2 # No of attributes
    attns = AttentionRaw((N, task2vec_dim)) #AttentionRawList(task2vec_dim, N)  
    n_epochs = 4000
    gamma = 1.
    lr = 1e-1
    optimizer = torch.optim.Adam(attns.parameters(), lr = lr)
    for epoch in range(n_epochs):
        pos_loss, neg_loss, total_loss = train(attns, Z_tensor, optimizer, gamma = gamma )
        if epoch%200 == 0:
            print(f"Epoch: {epoch}, pos_loss: {pos_loss}, neg_loss: {neg_loss}, total_loss: {total_loss}")

    
    attrs = attns(Z_tensor)
    task_attributes = attrs.transpose(0,1)
    task_attribute_magnitudes = task_attributes.sum(-1)
    data = pd.DataFrame(task_attribute_magnitudes,  columns =['x','y'])
    data['lang'] = ref['lang']
    data['attr'] = ref['attr']
    data['name'] = ref['name']
    import seaborn as sns
    sns.set(rc={'figure.figsize':(15,15)})
    sns_plot = sns.scatterplot(data=data, x="x", y="y", hue = "attr", style="lang", s = 300)
    t = datetime.datetime.now()
    name = t.strftime('%I_%M%p_%B%d')
    sns_plot.get_figure().savefig(f"viz/soft-{n_epochs}-{lr}-{name}.png", dpi = 400)