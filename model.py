import torch
import torch.nn as nn
import torch_geometric
import torch.fft as fft
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
from MODEL.NAM import LinReLU, ExU
from torch.distributions.normal import Normal
import tensorly
from torch.nn import MultiheadAttention
from einops import rearrange

import torch
import torch.nn as nn
from MODEL.TNAM import *



def freq_mix(x, rank=2, rate=0.5, dim=1):

    x_f = torch.fft.fft(x, dim=dim)

    m = torch.FloatTensor(x_f.shape).to(x.device).uniform_() < rate
    amp = abs(x_f)
    _, index = amp.sort(dim=dim, descending=True)
    dominant_mask = index > rank

    m = torch.bitwise_and(m, dominant_mask)
    freal = x_f.real.masked_fill(m, 0)
    fimag = x_f.imag.masked_fill(m, 0)

    b_idx = torch.randperm(x.shape[0])
    x2 = x[b_idx]
    x2_f = torch.fft.fft(x2, dim=dim)
    m = torch.bitwise_not(m)
    freal2 = x2_f.real.masked_fill(m, 0)
    fimag2 = x2_f.imag.masked_fill(m, 0)
    freal += freal2
    fimag += fimag2
    x_f = torch.complex(freal, fimag)
    x = torch.abs(torch.fft.ifft(x_f, dim=dim))

    return x



class FourierTransform(nn.Module):
    def __init__(self):
        super(FourierTransform, self).__init__()

    def forward(self, x):
        x = x.unsqueeze(-1)
        x_complex = torch.cat((x, torch.zeros_like(x)), dim=-1)
        x_complex = torch.view_as_complex(x_complex)
        x_complex = x_complex.to(torch.complex64)
        x_fft = fft.fftn(x_complex, dim=-2)  
        return x_fft.to(torch.float32)



class VIGNN_TCL(nn.Module):
    def  __init__(self, input_dim, hidden_dim, num_layers=1, rank=1, ratio=0.5, dropout_rate=0, t=6):
        super(VIGNN_TCL, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rank = rank
        self.ratio = ratio

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
        self.weight1_var = nn.Parameter(torch.randn(input_dim, 1))
        self.weight2_var = nn.Parameter(torch.randn(1, input_dim))

        self.fc = nn.Linear(hidden_dim, 1)
        self.fc_att = nn.Linear(hidden_dim, 1)
        self.fc_var = nn.Linear(input_dim, hidden_dim)
        self.fc_v = nn.Linear(6, hidden_dim)

        self.fourier_transform = freq_mix
        self.weight = nn.Parameter(torch.randn(1, hidden_dim))
        self.weights = nn.Parameter(torch.randn(input_dim, hidden_dim, t))
        self.biases = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.layernorm = nn.LayerNorm(hidden_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, 0.0)

    def calculate_graph_laplacian(self, A):
        # A is the adjacency matrix with shape (num_nodes, num_nodes)
        num_nodes = A.shape[0]
        # Ensure A is of float type
        A = A.float()
        # Degree matrix D
        D = torch.sum(A, dim=1)  # shape: (num_nodes,)
        D_inv_sqrt = torch.where(D > 0, 1.0 / torch.sqrt(D), torch.zeros_like(D))
        # Identity matrix
        I = torch.eye(num_nodes, device=A.device)
        # Calculate normalized Laplacian matrix L = I - D^(-1/2) A D^(-1/2)
        L = I - torch.diag(D_inv_sqrt) @ A @ torch.diag(D_inv_sqrt)

        return L

    def forward(self,x):

        x = torch.flip(x, [1]) # b,t,v
        batch = x.size(0)
        time = x.size(1)
        num_var = x.size(-1)

        # Prepare for batch processing
        x_reshaped = x.permute(0, 2, 1)  # Shape: (B, num_var, T)
        x_var_t = torch.einsum('bvt,vht->bvh', x_reshaped, self.weights) + self.biases.unsqueeze(0)
        x_var_t = torch.relu(x_var_t)

        adj_var = torch.matmul(self.weight1_var, self.weight2_var)
        adj_var = torch.softmax(adj_var, dim=1)
        adj_var_weight = adj_var
        adj_var = adj_var + torch.eye(adj_var.shape[0]).to(x.device)
        adj_var = adj_var.unsqueeze(0).repeat(batch, 1, 1)
        graph_out = adj_var @ x_var_t
        graph_out = self.fc_v(graph_out)

        h0 = torch.zeros(self.num_layers, batch, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch, self.hidden_dim).to(x.device)

        x_fft = self.fourier_transform(x, rank=self.rank, rate=self.ratio, dim=1)

        x_var_t , _ = self.lstm(x, (h0, c0)) #B,T,V
        x_var_fft , _ = self.lstm(x_fft, (h0, c0)) # B,T,V

        x_var_t = x_var_t[:,-1,:]

        lstm_out = x_var_fft + graph_out

        lstm_out = self.dropout(lstm_out)
        lstm_out = self.fc(lstm_out)
        lstm_out = self.sigmoid(lstm_out)

        return lstm_out,  x_var_t,  x_var_fft, adj_var_weight

class CLoss(nn.Module):
    def __init__(self,):
        super(CLoss, self).__init__()

    def contrastive_loss(self, z1, z2, temp=1, eps=1e-8):
        B,T,D = z1.size()
        uni_z1 = z1[torch.randperm(z1.shape[0]), :, :].view(z1.size())
        uni_z2 = z2[torch.randperm(z2.shape[0]), :, :].view(z2.size())
        z = torch.cat([z1, z2, uni_z1, uni_z2], dim=0)
        z = z.transpose(0, 1)  # T x 2B x C
        sim = torch.matmul(z[:, : 2 * B, :], z.transpose(1, 2)) / temp  # T x 2B x 2B
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)
        logits = logits[:, :2 * B, :(2 * B - 1)]
        i = torch.arange(B, device=z1.device)
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        return loss

    def forward(self,z1, z2):
        d = 0
        loss = torch.tensor(0.0).to(z1.device)
        while z1.size(1) > 1:
            d += 1
            loss += self.instance_contrastive_loss_mixup(z1, z2)
            z1 = F.max_pool1d(z1.transpose(1,2), kernel_size=2).transpose(1,2)
            z2 = F.max_pool1d(z2.transpose(1,2), kernel_size=2).transpose(1,2)

        return loss

























