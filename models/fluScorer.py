import sys
import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
from torch import _VF, Tensor
from torch.nn.init import xavier_uniform_
from transformers import AutoModel, AutoConfig, AutoTokenizer
from models.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling1,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)

from nets_utils import (
    get_activation,
    make_pad_mask,
    trim_by_ctc_posterior,
)

#Attention pooling
class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, 1),
            nn.GELU(),
        )

    def forward(self, x, attn, mask):
        w = self.attention(attn).float()
        w[mask==0] = float('-inf')
        w = torch.softmax(w, 1)
        x = torch.sum(w * x, dim=1)
        return x

class StatPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.proj_layer = nn.Linear(2 * in_dim, in_dim)

    def forward(self, x, mask):
        input_mask_expanded = (
            mask.unsqueeze(-1).expand(x.size()).float()
        )
        mean = torch.sum(x * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
        variance = torch.sum(((x - mean.unsqueeze(1)) ** 2) * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        std_dev = torch.sqrt(variance)
        
        pooled = torch.cat((mean, std_dev), dim=1)
        stat_x = self.proj_layer(pooled)
        return stat_x

#Mean pooling
class MeanPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

    def forward(self, x, mask):
        input_mask_expanded = (
            mask.unsqueeze(-1).expand(x.size()).float()
        )
        return torch.sum(x * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

def create_mask(feature_embedding, seq_lengths):
    device = feature_embedding.device
    B, T, D = feature_embedding.shape
    range_tensor_for_mask = torch.arange(T).expand(B, T).to(device)
    # 1D mask (B, T)
    mask_1d = range_tensor_for_mask < seq_lengths.unsqueeze(1)
    # 2D mask (B, T, D)
    mask_2d = mask_1d.unsqueeze(2).expand(B, T, D) 
    
    return mask_1d, mask_2d

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def _reset_parameters(self):
        xavier_uniform_(self.qkv.weight, gain=1 / math.sqrt(2))

    def split_into_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute([0, 2, 1, 3])

    def forward(self, x, mask=None):
        B, N, C = x.shape

        #print(C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(
                mask.unsqueeze(1).unsqueeze(2),
                -1e9,
            )

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., rnn=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # NOTE: attention
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        # attention
        x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# adapt: tanh -> GELU
class BiLSTMScorer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, pred: str = "linear"):
        '''
        BiLSTM(input_size, hidden_size, num_layers=num_layers)
        '''
        super().__init__()
        self.blstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bias=True, batch_first=True, bidirectional=True)
        self.stat_pool = StatPooling(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, 1)
    
    def forward(self, x, act=None, seq_lengths=None):
        x_nopadded = pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True)
        output, hidden = self.blstm(x_nopadded)
        BiLSTM_embedding, out_len = pad_packed_sequence(output, batch_first=True)        
        mask_1d, mask_2d = create_mask(BiLSTM_embedding, seq_lengths)

        output = self.stat_pool(BiLSTM_embedding, mask_1d)

        score = self.fc(output)
        
        return score

class TfrScorer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 3, num_heads: int = 1, pred: str = "linear"):
        '''
        Note: input_size == hidden_size
        '''
        super().__init__()
        #self.projLayer = nn.Linear(input_size, hidden_size)
        self.blocks = nn.ModuleList([ Block(dim=input_size, num_heads=num_heads, attn_drop=0., drop=0.) for _ in range(num_layers)])
        self.fc = nn.Linear(input_size, 1)
    
    def forward(self, x, act=None, seq_lengths=None):
        mask_1d, mask_2d = create_mask(x, seq_lengths)
        mask_1d = mask_1d == 0
        
        for block in self.blocks:
            x = block(x=x, mask=mask_1d)
        
        tfr_embedding = x
        output = mean_pooling(tfr_embedding, mask_2d)

        score = self.fc(output)
        
        return score

def _preprocessing(audio_embedding, preprocessing_module):
    audio_embedding_list = []
    for i in range(audio_embedding.size(0)):
        new_audio_embedding = preprocessing_module(audio_embedding[i])
        audio_embedding_list.append(new_audio_embedding)
    new_audio_embedding_tensor = torch.stack(audio_embedding_list, dim=0)
    return new_audio_embedding_tensor

class FluencyScorerNoclu(nn.Module):
    ''' 
        A model for fluency score prediction without using cluster.
    '''
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.preprocessing = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Dropout(p=0.5),
            # nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Tanh()
        )
        self.scorer = BiLSTMScorer(embed_dim+clustering_dim, embed_dim, 2)

    def forward(self, x):
        ''' 
        x: extract audio features
        return: a pred score
        '''
        # step 1: audio features preprocessing
        new_audio_embedding_tensor = _preprocessing(x, self.preprocessing)

        # step 2: make a score directly
        pred = self.scorer(new_audio_embedding_tensor)
        return pred

class FluencyScorer(nn.Module):
    ''' 
        The main model for fluency score prediction with using cluster.
    '''
    def __init__(self, input_dim, embed_dim, clustering_dim=6, scorer="bilstm"):
        super().__init__()
        self.preprocessing = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Tanh()
        )
        self.cluster_embed = nn.Embedding(50 + 1, clustering_dim, padding_idx=0)
        
        if scorer == "bilstm":
            self.scorer = BiLSTMScorer(input_size=embed_dim+clustering_dim, hidden_size=embed_dim, num_layers=2)
        elif scorer == "tfr":
            self.scorer = TfrScorer(input_size=embed_dim+clustering_dim, hidden_size=embed_dim, num_layers=1)

    def forward(self, x, cluster_id):
        ''' 
        x: extract audio features
        return: a pred score
        '''
        device = x.device
        # step 1: audio features preprocessing
        nonzero_mask = x.abs().sum(dim=2) != 0
        seq_lengths = nonzero_mask.sum(dim=1).to(device)
        #new_audio_embedding_tensor = _preprocessing(x, self.preprocessing)
        new_audio_embedding_tensor = self.preprocessing(x)
        cluster_embed = self.cluster_embed(cluster_id).float()
        
        # step 2: concat audio and cluster embedding
        audio_features = torch.concat((new_audio_embedding_tensor, cluster_embed), dim=-1)

        # create mask
        mask_1d, mask_2d = create_mask(audio_features, seq_lengths)
        
        audio_features = audio_features * mask_2d

        # step 3: make a score
        pred = self.scorer(x=audio_features, seq_lengths=seq_lengths)
        
        return pred

