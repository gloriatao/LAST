import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

logger = logging.getLogger(__name__)

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, block_size, resid_pdrop, attn_pdrop,  n_head, band_width):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.band_width = band_width
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        for i in range(self.mask.shape[2]):
            r = max(i-self.band_width, 0)
            self.mask[:,:,i,:r] = 0
        self.n_head = n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, block_size, resid_pdrop, attn_pdrop,  n_head, band_width):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, block_size, resid_pdrop, attn_pdrop,  n_head, band_width)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_embd*2),
            nn.GELU(),
            nn.Linear(n_embd*2, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

def create_sinusoidal_embeddings(n_pos, dim, out):
    out.detach_()
    out.requires_grad = False
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, n_embd, n_layer, n_head, band_width=[1000-1, 10-1], block_size=7000, attn_pdrop=0.1, embd_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        # input embedding stem
        self.band_width = band_width
        self.pos_emb = nn.Embedding(7000, n_embd)
        self.position_enc = create_sinusoidal_embeddings(n_pos=7000, dim=n_embd, out=self.pos_emb.weight)
        self.drop = nn.Dropout(embd_pdrop)
        # transformer
        self.blocks_phase = nn.Sequential(*[Block(n_embd, block_size, resid_pdrop, attn_pdrop,  n_head, self.band_width[0]) for _ in range(n_layer)])
        self.blocks_tool = nn.Sequential(*[Block(n_embd, block_size, resid_pdrop, attn_pdrop,  n_head, self.band_width[1]) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.block_size = block_size
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        b, seq_len, seq_dim = x.size()
        position_ids_absolute = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_embeddings_absolute = self.pos_emb(position_ids_absolute)
        x = self.drop(x + position_embeddings_absolute)
        x_phase = self.ln_f(self.blocks_phase(x))
        x_tool = self.ln_f(self.blocks_tool(x))
        return x_phase, x_tool

