import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
torch.cuda.is_available()
from torch_geometric.nn import GENConv, GCNConv, HypergraphConv
from torch_geometric.nn import global_max_pool as gmp, global_add_pool as gap
from math import sqrt
import math

class Emden(torch.nn.Module):
    def __init__(self, n_output=2, num_features_xd=78,num_features_xv=3904,num_features_xf=881,num_features_xs=61,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.01):

        super(Emden, self).__init__()

        # smile branch
        self.n_output = n_output
        self.conv1 = HypergraphConv(num_features_xd, num_features_xd, use_attention=False)
        self.conv2 = HypergraphConv(num_features_xd, num_features_xd*4, use_attention=False)
        self.conv3 = GCNConv(num_features_xd*4, num_features_xd*10)
        self.fc_g1 = nn.Linear(num_features_xd*10*2, 1500)
        self.fc_g2 = nn.Linear(1500, output_dim)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax()
        #self.softmax = nn.functional.softmax()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.n1024 = nn.BatchNorm1d(1024)
        self.n256 = nn.BatchNorm1d(256)
        self.n128 = nn.BatchNorm1d(128)
        self.nxv = nn.BatchNorm1d(output_dim*6)
        self.nxf = nn.BatchNorm1d(num_features_xf)
        self.nxs = nn.BatchNorm1d(num_features_xs)

        # 1D fingerprint (881, transformer encoder) num_hidden_layers, ori_feature_dim, embed_dim, num_heads, middle_dim
        self.trans_f = TransformerEncoder(1, 1, 64, 8, 64, 1) 
        self.flat_fc = nn.Linear(num_features_xf*64, num_features_xf)
        #self.fc_f = nn.Linear(num_features_xf, 256) # test FC of fingerprint

        # 1D protein sequence before (61*20, transformer encoder)
        self.trans_xsb = TransformerEncoder(1, 20, 20, 4, 32, 1)
        self.flat_xs = nn.Linear(num_features_xs*20, num_features_xs)

        # 1D protein sequence after (61*20, transformer encoder)
        self.trans_xsa = TransformerEncoder(1, 20, 20, 4, 32, 1)

        # 1D protein features (hhm profiles, secondary structure, rASA)
        self.fc_xv = nn.Linear(num_features_xv, output_dim*6)

        # FC layers
        self.fc2_1 = nn.Linear(num_features_xf + output_dim*6, 1024)
        self.fc2_2 = nn.Linear(num_features_xs*2, 128)
        self.fc_3 = nn.Linear(output_dim + 1024 + 128, 256)
        self.fc_4 = nn.Linear(256, 128)
        self.fc_5 = nn.Linear(128, 32)
        self.out = nn.Linear(32, self.n_output)        # n_output = 2 for CrossEntropyLoss (https://blog.csdn.net/Penta_Kill_5/article/details/118085718)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        fingerprint = data.fingerprint.float()
        seqbefore = data.seqbefore.float()
        seqafter = data.seqafter.float()
        variant = data.variant.float()
        # print('x shape = ', x.shape)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)

        # first layers
        fingerprint = fingerprint.unsqueeze(-1)
        trans_f = self.trans_f(fingerprint)
        flatten_trans_f = torch.flatten(trans_f, start_dim=1, end_dim=2)
        flat_fc = self.flat_fc(flatten_trans_f)
        flat_fc = self.relu(flat_fc)
        flat_fc = self.nxf(flat_fc)
        trans_sb = self.trans_xsb(seqbefore)
        trans_sa = self.trans_xsa(seqafter)
        flatten_trans_sb = torch.flatten(trans_sb, start_dim=1, end_dim=2)
        flat_xsb = self.flat_xs(flatten_trans_sb)
        flat_xsb = self.relu(flat_xsb)
        #flat_xsb = self.nxs(flat_xsb)
        flatten_trans_sa = torch.flatten(trans_sa, start_dim=1, end_dim=2)
        flat_xsa = self.flat_xs(flatten_trans_sa)
        flat_xsa = self.relu(flat_xsa)
        #flat_xsa = self.nxs(flat_xsa)
        fc_v = self.fc_xv(variant)
        fc_v = self.relu(fc_v)
        fc_v = self.dropout(fc_v)
        fc_v = self.nxv(fc_v)

        # flatten
        #xf = fc_f.view(-1, 32 * 121)

        # first concat
        concat1 = torch.cat((flat_fc, fc_v), 1)
        concat2 = torch.cat((flat_xsb, flat_xsa), 1)

        # second layers
        fc2_1 = self.fc2_1(concat1)
        fc2_1 = self.relu(fc2_1)
        fc2_1 = self.dropout(fc2_1)
        fc2_1 = self.n1024(fc2_1)
        fc2_2 = self.fc2_2(concat2)
        fc2_2= self.relu(fc2_2)
        fc2_2 = self.n128(fc2_2)

        # merge
        concat3 = torch.cat((fc2_1, fc2_2, x), 1)

        # last layers
        xc = self.fc_3(concat3)
        xc = self.relu(xc)
        xc = self.fc_4(xc)
        xc = self.relu(xc)
        xc = self.fc_5(xc)
        xc = self.relu(xc)
        xc = self.out(xc)
        
        #return nn.functional.log_softmax(xc)
        return xc

def scaled_dot_product_attention(query, key, value, query_mask=None, key_mask=None, mask=None):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    if query_mask is not None and key_mask is not None:
        mask = torch.bmm(query_mask.unsqueeze(-1), key_mask.unsqueeze(1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float("inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None):
        attn_outputs = scaled_dot_product_attention(
            self.q(query), self.k(key), self.v(value), query_mask, key_mask, mask)
        return attn_outputs

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None):
        x = torch.cat([
            h(query, key, value, query_mask, key_mask, mask) for h in self.heads
        ], dim=-1)
        x = self.output_linear(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, embed_dim, middle_dim):
        super().__init__()
        self.linear_1 = nn.Linear(embed_dim, middle_dim)
        self.linear_2 = nn.Linear(middle_dim, embed_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.01)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, middle_dim):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, middle_dim)

    def forward(self, x, mask=None):
        # Apply layer normalization and then copy input into query, key, value
        hidden_state = self.layer_norm_1(x)
        # Apply attention with a skip connection
        x = x + self.attention(hidden_state, hidden_state, hidden_state, mask=mask)
        # Apply feed-forward layer with a skip connection
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x
'''
class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout, max_len=1000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        div_term2 = torch.exp((torch.arange(0, dim-1, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term2)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.drop_out = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        print(emb.shape)
        emb = emb * math.sqrt(self.dim)
        print(emb.shape)
        print(self.pe.shape)
        print(self.pe[:emb.size(0)].shape)
        self.pe = self.pe.transpose(1, 2)
        print(self.pe[:emb.size(0)].shape)
        if step is None:
            emb = emb + self.pe[:emb.size(0)].to(emb.device)
        else:
            emb = emb + self.pe[step].to(emb.device)
        emb = self.drop_out(emb)
        print(emb.shape)
        return emb
'''

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        X2 = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens-1, 2, dtype=torch.float32) / (num_hiddens-1))
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X2)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class Embeddings(nn.Module):
    def __init__(self, ori_feature_dim, embed_dim):
        super().__init__()
        # self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.token_embeddings = nn.Linear(ori_feature_dim, embed_dim) # change embedding into linear for onehot
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-12)
        self.dropout = nn.Dropout(0.02)

    def forward(self, inputs, input_dim):
        #inputs = torch.tensor(inputs).to(inputs.device).long()
        # Create token and position embeddings
        token_embeddings = self.token_embeddings(inputs)
        PE = PositionalEncoding(input_dim, 0)
        position_embeddings = PE(inputs)
        # Combine token and position embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class TransformerEncoder(nn.Module):
    def __init__(self, num_hidden_layers, vocab_size, embed_dim, num_heads, middle_dim, input_dim):
        super().__init__()
        self.embeddings = Embeddings(vocab_size, embed_dim)
        self.dim = input_dim
        self.layers = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads, middle_dim)
                                     for _ in range(num_hidden_layers)])

    def forward(self, x, mask=None):
        x = self.embeddings(x, self.dim)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x