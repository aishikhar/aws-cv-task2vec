from task2vec import Task2Vec
from models import get_model
import datasets
import task_similarity
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as tt
from tqdm import tqdm

from synbols_utils import Synbols

def _similarity_matrix(attrs, epsilon = 1e-8):
    norm = attrs.norm(dim = -1, keepdim = True)
    norm = torch.maximum(norm, 1e-8 * torch.ones(norm.shape))
    attrs = attrs / norm
    temp = attrs.view(attrs.shape[0], attrs.shape[1],1,1,attrs.shape[2])
    similarity_matrix = torch.mul(temp,attrs).sum(axis = -1)
    return similarity_matrix

def cos_sim(A,B, eps=1e-8, ):
    A_norm = A.norm(dim = -1, keepdim = True)
    B_norm = B.norm(dim = -1, keepdim = True)
    A_norm = torch.maximum(A_norm, eps * torch.ones(A_norm.shape))
    B_norm = torch.maximum(B_norm, eps * torch.ones(B_norm.shape))
    A = torch.div(A,A_norm)
    B = torch.div(B,B_norm)
    return torch.mm(A,B.T)
#cos_sim(Attrs[0],Attrs[0]).flatten().contiguous()

def norm_unitvectors(M, eps = 1e-8):
    M_norm = M.norm(dim = -1, keepdim = True)
    M_norm = torch.maximum(M_norm, eps * torch.ones(M_norm.shape))
    return torch.div(M, M_norm)
def positives(Attrs):
    M = norm_unitvectors(Attrs)
    M_trans = M.transpose(1,2)
    return torch.bmm(M, M_trans)

def positive_loss(Attrs):
   x = -torch.log(positives(Attrs).flatten())
   return x.sum()