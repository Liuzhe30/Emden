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
                                                                           
def softmax(vec):
    """Compute the softmax in a numerically stable way."""
    vec = vec - np.max(vec)  # softmax(x) = softmax(x+c)
    exp_x = np.exp(vec)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_probs = torch.Tensor()
    total_raw_preds = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            #n = data.y.size(0)
            #label = torch.zeros(n,2)
            #label = label.scatter_(dim=1, index=data.y.cpu().view(-1, 1), src=torch.ones(n, 2))
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.cpu()), 0)
            total_probs = torch.cat((total_probs, output.cpu()), 0)
            total_raw_preds = torch.cat((total_raw_preds, output.cpu()), 0)
    
    total_probs = total_probs.numpy()
    #total_labels = total_labels.argmax(dim=1).numpy()
    total_preds = total_preds.argmax(dim=1).numpy()
    
    for i in range(total_probs.shape[0]):
        total_probs[i] = softmax(total_probs[i])
    total_probs = total_probs[:,1]
    
    #print(total_raw_preds)
    #print(total_labels)
    #print(total_preds)
    return total_labels.flatten().numpy(),total_preds.flatten(),total_probs.flatten(),total_raw_preds.numpy() 

def generate_pred(weipath,savepath,root,dataset):

    cuda_name = "cuda:0"
    if len(sys.argv)>3:
        cuda_name = ["cuda:0","cuda:1"][int(sys.argv[3])]
    print('cuda_name:', cuda_name)
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    #processed_data_file_test = '../datasets/processed/test_data.pt'
    test_data = PyDataset(root=root, dataset=dataset)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)


    model_file_name = weipath + 'weights.model'
    model = Emden().to(device)
    model.load_state_dict(torch.load(model_file_name))
    print('predicting for test data')
    G,P,T,R = predicting(model, device, test_loader)
    #print(G.shape,P.shape,T.shape,R.shape)
    np.save(savepath + 'label.npy',G)
    np.save(savepath + 'pred.npy',P)
    np.save(savepath + 'pred_prob.npy',T)
    np.save(savepath + 'pred_raw.npy',R)

if __name__=='__main__':
    
    #'''
    # for real performance calculation
    generate_pred('../model/weights/', '../model/pred_results/ori/','../datasets','test')
    generate_pred('../model/weights/', '../model/pred_results/rev/','../datasets','da_rev_test') 
    #'''

    # for external evaluation
    generate_pred('../model/weights/', '../model/pred_results/drsp/','../datasets','DRSP')
