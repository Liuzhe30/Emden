import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.getcwd())
from random import shuffle
import torch
import torch.nn as nn
import json
from src.utils import PyDataset, cee, get_acc
from sklearn.metrics import f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from model.emden import Emden
from torch_geometric.data import InMemoryDataset, DataLoader

# training function at each epoch
def train_emden(model, device, train_loader, optimizer, epoch, loss_fn, LOG_INTERVAL):
    print('Training on {} samples...'.format(len(train_loader.dataset)))

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        #print(output)
        loss.backward()
        #print([x.grad for x in optimizer.param_groups[0]['params']])
        optimizer.step()
        #print(optimizer.param_groups[0])
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tACC: {:.3f}'.format(epoch,
                                                                           batch_idx,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item(),
                                                                           get_acc(output, data.y.view(-1, 1).float())))
                                                                           

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1)), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()

def train_5fold():
    cuda_name = "cuda:0"
    if len(sys.argv)>3:
        cuda_name = ["cuda:0","cuda:1"][int(sys.argv[3])]
    print('cuda_name:', cuda_name)

    TRAIN_BATCH_SIZE = 60
    VALID_BATCH_SIZE = 256
    TEST_BATCH_SIZE = 256
    LR = 0.05
    LOG_INTERVAL = 5
    NUM_EPOCHS = 100

    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)

    fold = 1
    processed_data_file_train = '../datasets/fivefold/processed/' + str(fold) + 'fold_train_data.pt'
    processed_data_file_valid = '../datasets/fivefold/processed/' + str(fold) + 'fold_valid_data.pt'
    processed_data_file_test = '../datasets/processed/test_data.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_valid)) or (not os.path.isfile(processed_data_file_test))):
        print('please run prepareData.py to prepare data in pytorch format!')
    else:
        train_data = PyDataset(root='../datasets/fivefold', dataset=str(fold)+'fold_train')
        valid_data = PyDataset(root='../datasets/fivefold', dataset=str(fold)+'fold_valid')
        test_data = PyDataset(root='../datasets', dataset='test')
        print(train_data)
        print(valid_data)
        print(test_data)

        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=VALID_BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = Emden().to(device)
        print(model)
        loss_fn = nn.CrossEntropyLoss() # nn.logSoftmax() and nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_cee = 1000
        best_valid_cee = 1000
        best_epoch = -1
        model_file_name = '../model/weights/' + str(fold) + 'fold_weights.model'

        for epoch in range(NUM_EPOCHS):
            train_emden(model, device, train_loader, optimizer, epoch+1, loss_fn, LOG_INTERVAL)
            # print('predicting for valid data')

        '''
        for epoch in range(NUM_EPOCHS):
            train_emden(model, device, train_loader, optimizer, epoch+1, loss_fn, LOG_INTERVAL)
            print('predicting for valid data')
            G,P = predicting(model, device, valid_loader)
            #print(G) # int labels
            print(P) # probability
            P = P.astype(np.int64)
            print(P)
            val = cee(G,P)
            ret = [epoch, cee(G,P),f1_score(G,P),roc_auc_score(G,P)]
            if val < best_cee:
                best_cee = val
                best_epoch = epoch+1
                torch.save(model.state_dict(), model_file_name)
                print('predicting for test data')
                G,P = predicting(model, device, test_loader)
                print('loss decreased at epoch ', best_epoch, '; best_test_f1, best_test_auc:', ret[2], ret[3])
            else:
                print(ret[1],'No improvement since epoch ', best_epoch, '; best_test_f1, best_test_auc:', ret[2], ret[3])
            '''

def train():
    pass

if __name__=='__main__':
    train_5fold()