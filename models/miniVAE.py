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

    def __init__(self, n_embd, block_size, resid_pdrop, attn_pdrop,  n_head):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
        self.n_head = n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, 0)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, block_size, resid_pdrop, attn_pdrop,  n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, block_size, resid_pdrop, attn_pdrop,  n_head)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
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


class AverageSelfAttention(nn.Module):
    def __init__(self, attention_size):
        super(AverageSelfAttention, self).__init__()
        w = torch.empty(attention_size)
        nn.init.normal_(w, std=0.02)
        self.attention_weights = nn.Parameter(w)
        self.softmax = nn.Softmax(dim=-1)
        self.non_linearity = nn.GELU()

    def forward(self, inputs, attention_mask=None):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.non_linearity(inputs.matmul(self.attention_weights))

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        if attention_mask is not None:
            scores = scores + attention_mask

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################
        scores = self.softmax(scores)

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        representations = weighted.sum(1).squeeze(1)

        return representations, scores

class encoder(nn.Module):
    def __init__(self, n_embd, n_layer, n_head, block_size=8000, attn_pdrop=0.1, resid_pdrop=0.1,):
        super(encoder, self).__init__()
        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, block_size, resid_pdrop, attn_pdrop,  n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.block_size = block_size

    def forward(self, x):
        x = self.blocks(x)
        x = self.ln_f(x)
        return x

class vae(nn.Module):
    def __init__(self, n_embd, n_layer, n_head, block_size=8000, attn_pdrop=0.1, embd_pdrop=0.1, resid_pdrop=0.1,learn_prior=True):
        super(vae, self).__init__()
        # input embedding stem
        self.learn_prior = learn_prior
        self.pos_emb = nn.Embedding(8000, n_embd)
        self.position_enc = create_sinusoidal_embeddings(n_pos=8000, dim=n_embd, out=self.pos_emb.weight)
        self.drop = nn.Dropout(embd_pdrop)

        self.encoder = encoder(n_embd, n_layer, n_head, block_size=block_size, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop)
        self.posterior_averageSelfAttention = AverageSelfAttention(n_embd)
        self.posterior_mean = nn.Linear(n_embd, n_embd, bias=False)
        self.posterior_logvar = nn.Linear(n_embd, n_embd, bias=False)

        if learn_prior:
            self.encoder_prior = encoder(n_embd, n_layer, n_head, block_size=block_size, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop)
            self.prior_averageSelfAttention = AverageSelfAttention(n_embd)
            self.prior_mean = nn.Linear(n_embd, n_embd, bias=False)
            self.prior_logvar = nn.Linear(n_embd, n_embd, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def reparameterize(self, mean, logvar, z=None):
        std = logvar.mul(0.5).exp()
        if z is None:
            z = torch.randn(std.size(), device=mean.device, dtype=mean.dtype)
        return z.mul(std) + mean

    def kl_loss(self, mean1, logvar1, mean2, logvar2):
        exponential = logvar1 - logvar2 - torch.pow(mean1 - mean2, 2) / logvar2.exp() - torch.exp(logvar1 - logvar2) + 1
        result = -0.5 * torch.sum(exponential, tuple(range(1, len(exponential.shape))))
        return result.mean()

    def forward(self, x, y, from_prior=True, from_mean=False, return_gaussian=False):
        # latent representation
        b, seq_len, seq_dim = x.size()
        position_ids_absolute = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_embeddings_absolute = self.pos_emb(position_ids_absolute)
        x = self.drop(x + position_embeddings_absolute)
        x = self.encoder(x)
        representations, _ = self.posterior_averageSelfAttention(x)
        posterior_mean = self.posterior_mean(representations)
        posterior_logvar = self.posterior_logvar(representations)

        if self.learn_prior:
            y = self.drop(y + position_embeddings_absolute)
            y = self.encoder(y)
            # y = self.encoder_prior(y)
            representations, _ = self.prior_averageSelfAttention(y)
            prior_mean = self.prior_mean(representations)
            prior_logvar = self.prior_logvar(representations)
        else:
            prior_mean = prior_logvar = torch.zeros([x.size(0), self.config.n_embd], device=x.device)
            prior_mean, prior_logvar = prior_mean.to(posterior_mean.dtype), prior_logvar.to(posterior_logvar.dtype)

        if from_prior:
            latent_mean, latent_logvar = prior_mean, prior_logvar
        else:
            latent_mean, latent_logvar = posterior_mean, posterior_logvar

        if from_mean:
            z = latent_mean
        else:
            z = self.reparameterize(latent_mean, latent_logvar)
        assert not torch.isnan(z).any(), 'training get nan z'

        kl_loss = self.kl_loss(posterior_mean, posterior_logvar, prior_mean, prior_logvar).unsqueeze(0)

        if return_gaussian:
            return z, kl_loss, posterior_mean, posterior_logvar, prior_mean, prior_logvar
        else:
            return z, kl_loss



