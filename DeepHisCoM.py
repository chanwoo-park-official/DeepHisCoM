import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import pandas as pd
import multiprocessing
import sys, os
import random
import time
import torch
from torch.multiprocessing import Pool, Process, set_start_method



def scenario_num_extract(path = os.path.realpath(__file__)):
    return str(path.split("/")[-2])

def rep_extract(scen_num, location):
    admin = open(loc + scen_num + '/simulation_administration0.txt', mode='r', encoding='utf-8')
    line = admin.readline()
    line = admin.readline()
    rep = str(line.split("\n")[0])
    admin.close()
    return rep

def perm_extract(scen_num, location):
    admin = open(loc + scen_num + '/simulation_administration0.txt', mode='r', encoding='utf-8')
    line = admin.readline()
    line = admin.readline()
    line = admin.readline()
    line = admin.readline()
    line = admin.readline()
    line = admin.readline()
    perm = int(line.split("\n")[0])
    admin.close()
    return perm

def GPUnum_extract(scen_num, location):
    admin = open(loc + scen_num + '/GPU.txt', mode='r', encoding='utf-8')
    line = admin.readline()
    GPU = int(line.split("\n")[0])
    admin.close()
    return GPU


class CustomDataset(Dataset): 
    def __init__(self, DATA, inputlength):
        self.x_data = DATA[:,0:(inputlength-1)]
        self.y_data = DATA[:,(inputlength-1)].reshape(-1,1)     

    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y
  

# scenario = scenario_num_extract()                               
# loc = "/data/member/cwpark/liv/article_metrenew/withgpu/simulation/"
best_nconv = [2, 0, 2, 0, 0, 0, 2, 2, 2, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0]
#best_nconv = [0, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0, 2, 2, 0, 2, 0, 2]  We can use this conv(small biomarker number -> small nconv
# rep = rep_extract(scenario, loc)
seed = 210907
penalty_bio_path1 = 0
penalty_bio_path2 = 0
penalty_path_disease_list_log = range(-100, -1, 1)
penalty_path_disease_list = [pow(10, penalty_path_disease_list_log[i]/10) for i in range(len(penalty_path_disease_list_log))]
penalty_path_disease = 0.00015848
# total_permutation_num = perm_extract(scenario, loc)


class DeepHisCoM(nn.Module):
    def __init__(self,  nvar, width, device):
        super(DeepHisCoM, self).__init__()
        self.nvar = nvar
        self.width = width
        self.fc_bio_path1 = nn.ModuleList([nn.Linear(nvar[i], width[i], bias = False) for i in range(len(self.nvar))])
        self.dropout_bio_path1 = nn.Dropout(0.5)
        self.fc_bio_path2 = nn.ModuleList([nn.Linear(width[i], 1, bias = False) for i in range(len(self.nvar))])
        self.bn_path = nn.BatchNorm1d(len(nvar))
        self.dropout_path = nn.Dropout(0.5)
        self.fc_path_disease=nn.Linear(len(nvar),1)
        self.device = device

    def forward(self, x):
        kk=0
        s=list()
        for i, l in enumerate(self.fc_bio_path1):    
            k=kk
            kk=kk+self.nvar[i]
            t= x[:,k:kk]
            s.append(self.dropout_bio_path1(F.leaky_relu(l(t), 0.2)))

        for i, l in enumerate(self.fc_bio_path2):
            if i==0:
                if self.width[i] == 0:
                    x= s[i]
                else:
                    x=F.leaky_relu(l(s[i]), 0.2)
            else:
                if self.width[i] == 0:
                    x= torch.cat((x,s[i]),1)
                else:
                    x=torch.cat((x,F.leaky_relu(self.fc_bio_path2[i](s[i]), 0.2)),1)

        x = self.bn_path(x)
        x = x/(torch.norm(x,2))
        x = self.dropout_path(x)
        x = self.fc_path_disease(x)
        x = torch.sigmoid(x)
        return(x)


GPU_NUM = 2
is_cuda = torch.cuda.is_available()                                                                 
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)                                       # change allocation of current GPU
dtype = torch.FloatTensor
learning_rate = 0.01
'''
width_list = [best_width]
learning_rate_list = [best_learning_rate]
class_weight = torch.FloatTensor([1,1])                                             # if unbalanced data, we can change it
: For BCELoss
'''

total_permutation_num = 100000

for permutation in range(0, total_permutation_num):
    loss_save = []
    # print("start", permutation, "th permutation in secenario", scenario, ", replication", rep)
    print("start", permutation)
    torch.manual_seed(permutation); random.seed(permutation); np.random.seed(permutation)              # for permutation random seed
    train = pd.read_csv("/data/member/cwpark/liv/article_metrenew/tmpdata/perm0.csv")
    if permutation != 0:
        train['HCC'] = np.random.permutation(train['HCC'])                              # permute y-data
    train = torch.from_numpy(train.values).type(dtype)
    train_varlen = train.shape[1];    train_datanum = train.shape[0]
    data_train = CustomDataset(train, train_varlen)
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)  
    annot = pd.read_csv("/data/member/cwpark/liv/article_metrenew/annot_path.csv")     # annotation data
    groupunique = list(OrderedDict.fromkeys(annot["group"]))                            # nvar setting 
    grouplen = len(groupunique)
    nvar = []
    for i in range(0,grouplen):  
        nvar.append(sum(annot["group"]==groupunique[i]))


    data_train = DataLoader(data_train, batch_size = train_datanum, shuffle = True)
    model_DeepHisCoM = DeepHisCoM(nvar, best_nconv, device)
    if is_cuda:                                                                         # If we use GPU.
        model_DeepHisCoM.to('cuda')
    optimizer_DeepHisCoM = torch.optim.Adam([{'params': model_DeepHisCoM.parameters()}]
                                                , lr=learning_rate)                     # can change optimizer
    criterion = nn.BCELoss()
    count = 0
    count_lim = 10
    AUC_train = 0
    for __ in range(0,10000):
        for batch_idx, samples in enumerate(data_train):
            def closure():
                optimizer_DeepHisCoM.zero_grad()
                x_train, y_train = samples
                if is_cuda:                                                             # If we use GPU.
                    x_train = x_train.cuda()
                    y_train = y_train.cuda()
                output_DeepHisCoM = torch.squeeze(model_DeepHisCoM(x_train))
                y_train = torch.squeeze(y_train)
                loss_DeepHisCoM = (criterion(output_DeepHisCoM, y_train)).type(dtype)
                for param in model_DeepHisCoM.fc_path_disease.parameters():
                    loss_DeepHisCoM = loss_DeepHisCoM + penalty_path_disease * torch.norm(param,2)
                loss_DeepHisCoM=  loss_DeepHisCoM.type(dtype)
                loss_DeepHisCoM.backward()
                return(loss_DeepHisCoM)
            model_DeepHisCoM.train()
            loss = optimizer_DeepHisCoM.step(closure)
            model_DeepHisCoM.eval()
            optimizer_DeepHisCoM.zero_grad()
            x_train, y_train = samples
            if is_cuda:                                                             # If we use GPU.
                x_train = x_train.cuda()
                y_train = y_train.cuda()
            output_DeepHisCoM = torch.squeeze(model_DeepHisCoM(x_train))
            y_train = torch.squeeze(y_train)
            loss_DeepHisCoM = (criterion(output_DeepHisCoM, y_train)).type(dtype)
            loss_DeepHisCoM = (criterion(output_DeepHisCoM, y_train)).type(dtype)
            for param in model_DeepHisCoM.fc_path_disease.parameters():
                loss_DeepHisCoM = loss_DeepHisCoM + penalty_path_disease * torch.norm(param,2)  
            loss_DeepHisCoM=  loss_DeepHisCoM.type(dtype)
            loss_save.append(loss_DeepHisCoM)
            if min(loss_save) == loss_DeepHisCoM:
                count = 0
                param_save = np.array(list(model_DeepHisCoM.fc_path_disease.parameters())[0].tolist()[0])
                if torch.sum(torch.isnan(output_DeepHisCoM))==0:
                    AUC_train = roc_auc_score(y_train.cpu().detach().numpy(), output_DeepHisCoM.cpu().detach().numpy())
                else:
                    AUC_train = 0        
            else:
                count = count + 1
    

        if(count > count_lim):
            break


    param = param_save
    np.savetxt("/data/member/cwpark/liv/article_metrenew/withgpu/real_data/d2/param"+ str(permutation) + ".txt" ,param)



#if __name__ == '__main__':
#    set_start_method('spawn')
#    pool = torch.multiprocessing.Pool(processes = multiprocess_num)
#    pool.map(experiment, range(0,total_permutation_num))
#    pool.close()
#    pool.join()
