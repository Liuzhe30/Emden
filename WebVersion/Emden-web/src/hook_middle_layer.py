import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.getcwd())
from random import shuffle
import torch
import torch.nn as nn
import json
from src.utils import PyDataset, cee, get_acc
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')
import random
from model.emden import Emden
from torch_geometric.data import InMemoryDataset, DataLoader

inps = []
outs = []

def layer_hook(module, inp, out):
    inps.append(inp[0].data.cpu().numpy())
    outs.append(out.data.cpu().numpy())

def middle_output_final(model,train_loader,device):

    for data in train_loader:
        data = data.to(device)

        hook = model.fc_5.register_forward_hook(layer_hook)
        output = model(data)
        
        hook.remove()

    save_out = np.array(outs)
    print(save_out.shape) # (600, 1, 32)
    np.save('../model/middle_output/fc5_output.npy',save_out)

def middle_output_trans_f(model,train_loader,device):
    for data in train_loader:
        data = data.to(device)

        hook = model.trans_f.register_forward_hook(layer_hook)
        output = model(data)
        
        hook.remove()

    save_out = np.array(outs)
    print(save_out.shape) # 
    np.save('../model/middle_output/trans_f_output.npy',save_out)

def middle_output_trans_xsb(model,train_loader,device):
    for data in train_loader:
        data = data.to(device)

        hook = model.trans_xsb.register_forward_hook(layer_hook)
        output = model(data)
        
        hook.remove()

    save_out = np.array(outs)
    print(save_out.shape) # 
    np.save('../model/middle_output/trans_xsb_output.npy',save_out)

def middle_output_trans_xsa(model,train_loader,device):
    for data in train_loader:
        data = data.to(device)

        hook = model.trans_xsa.register_forward_hook(layer_hook)
        output = model(data)
        
        hook.remove()

    save_out = np.array(outs)
    print(save_out.shape) # 
    np.save('../model/middle_output/trans_xsa_output.npy',save_out)

def input_gcn(model,train_loader,device):
    for data in train_loader:
        data = data.to(device)

        hook = model.conv1.register_forward_hook(layer_hook)
        output = model(data)
        
        hook.remove()

    save_inp = np.array(inps)
    print(save_inp.shape) # 
    np.save('../model/middle_output/input_gcn.npy',save_inp)

def input_trans_f(model,train_loader,device):
    for data in train_loader:
        data = data.to(device)

        hook = model.trans_f.register_forward_hook(layer_hook)
        output = model(data)
        
        hook.remove()

    save_inp = np.array(inps)
    print(save_inp.shape) # 
    np.save('../model/middle_output/input_trans_f.npy',save_inp)

def input_trans_xsa(model,train_loader,device):
    for data in train_loader:
        data = data.to(device)

        hook = model.trans_xsa.register_forward_hook(layer_hook)
        output = model(data)
        
        hook.remove()

    save_inp = np.array(inps)
    print(save_inp.shape) # 
    np.save('../model/middle_output/input_trans_xsa.npy',save_inp)

def input_trans_xsb(model,train_loader,device):
    for data in train_loader:
        data = data.to(device)

        hook = model.trans_xsb.register_forward_hook(layer_hook)
        output = model(data)
        
        hook.remove()

    save_inp = np.array(inps)
    print(save_inp.shape) # 
    np.save('../model/middle_output/input_trans_xsb.npy',save_inp)

def input_fc_xv(model,train_loader,device):
    for data in train_loader:
        data = data.to(device)

        hook = model.fc_xv.register_forward_hook(layer_hook)
        output = model(data)
        
        hook.remove()

    save_inp = np.array(inps)
    print(save_inp.shape) # 
    np.save('../model/middle_output/input_fc_xv.npy',save_inp)

if __name__=='__main__':
    cuda_name = "cuda:0"
    if len(sys.argv)>3:
        cuda_name = ["cuda:0","cuda:1"][int(sys.argv[3])]
    print('cuda_name:', cuda_name)
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

    #processed_data_file_test = '../datasets/processed/test_data.pt'
    train_data = PyDataset(root='../datasets', dataset='train')
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
    model_file_name = '../model/weights/' + 'weights.model'
    model = Emden().to(device)
    model.load_state_dict(torch.load(model_file_name))

    print(model)
    model.eval()

    # hook (!! run one line each time for initial inps[] and outs[])
    #middle_output_final(model,train_loader,device) # (600, 1, 32)
    #middle_output_trans_f(model,train_loader,device) # (600, 1, 881, 64)
    #middle_output_trans_xsb(model,train_loader,device) # (600, 1, 61, 20)
    #middle_output_trans_xsa(model,train_loader,device) # (600, 1, 61, 20)
    #input_gcn(model,train_loader,device) # (600,)
    #input_trans_f(model,train_loader,device) # (600, 1, 881, 1)
    #input_trans_xsa(model,train_loader,device) # (600, 1, 61, 20)
    #input_trans_xsb(model,train_loader,device) # (600, 1, 61, 20)
    #input_fc_xv(model,train_loader,device) # (600, 1, 3904)