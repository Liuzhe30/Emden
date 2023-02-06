import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
torch.cuda.is_available()
from torch_geometric.nn import GENConv, GCNConv, HypergraphConv
from torch_geometric.nn import global_max_pool as gmp, global_add_pool as gap

class Emden(torch.nn.Module):
    def __init__(self, n_output=2, num_features_xd=78, num_features_xf=881,num_features_xv=3904,num_features_xs=1220,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(Emden, self).__init__()

        # smile branch
        self.n_output = n_output
        self.conv1 = HypergraphConv(num_features_xd, num_features_xd, use_attention=False)
        self.conv2 = HypergraphConv(num_features_xd, num_features_xd*4, use_attention=False)
        self.conv3 = GCNConv(num_features_xd*4, num_features_xd*10)
        self.fc_g1 = torch.nn.Linear(num_features_xd*10*2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax()
        self.dropout = nn.Dropout(dropout)

        # 1D fingerprint
        self.fc_f = nn.Linear(num_features_xf, output_dim)

        # 1D protein sequence before (flattened)
        self.fc_xsb = nn.Linear(num_features_xs, output_dim)

        # 1D protein sequence after (flattened)
        self.fc_xsa = nn.Linear(num_features_xs, output_dim)

        # 1D protein features (hhm profiles, secondary structure, rASA)
        self.fc_xv = nn.Linear(num_features_xv, output_dim*3)

        # combined layers
        self.fc1 = nn.Linear(output_dim*7, 512)
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, self.n_output)        # n_output = 2 for CrossEntropyLoss (https://blog.csdn.net/Penta_Kill_5/article/details/118085718)

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

        # test FC
        fc_f = self.fc_f(fingerprint)
        fc_sb = self.fc_xsb(seqbefore)
        fc_sa = self.fc_xsa(seqafter)
        fc_v = self.fc_xv(variant)

        # flatten
        #xf = fc_f.view(-1, 32 * 121)

        # concat
        xc = torch.cat((x, fc_f, fc_sb, fc_sa, fc_v), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        #xc = self.softmax(xc)
        out = self.out(xc)
        return out