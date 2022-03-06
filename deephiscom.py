import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
import pandas as pd
import multiprocessing
import sys, os
import random
import time
import torch
from torch.multiprocessing import Pool, Process, set_start_method
import argparse
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



def parse_args():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--seed", type=int, default=100, help="Random seed")
    parser.add_argument(
        "--perm", type=int, default = 1000, help="The number of permutations"
    )
    parser.add_argument(
        "--gpu", type=int, default = "0", help="The number of GPU"
    )
    parser.add_argument(
        "--dir", type = str, help = "the base working directory"
    )
    parser.add_argument(
        "--activation", type =str, default = "leakyrelu", help = "the used activation function (tanh, relu, leakyrelu, identity)"
    )
    parser.add_argument(
        "--loss", type = str, default = "BCELoss", help = "the used loss function (BCELoss or MSELoss)" 
    )
    parser.add_argument(
        "--reg_type", type = str, default = "l1", help = "the used regularization function type(l1 or l2)"
    )
    parser.add_argument(
        "--reg_const_pathway_disease", type = float, default = 0, help = "regularization constant between pathways and phenotype")
    parser.add_argument(
        "--reg_const_bio_pathway", type = float, default = 0, help = "regularization constant between biological factors and pathway")
    parser.add_argument(
        "--leakyrelu_const", type = float, default = 0.2, help = "the constants for leakyrelu"
    )
    parser.add_argument(
        "--dropout_rate", type = float, default = 0.5, help = "the constants for dropout_rate"
    )
    parser.add_argument(
        "--experiment_name", type=str, default="exp", help="A folder name (relative path) for saving results"
    )
    parser.add_argument(
        "--batch_size", type=int, default="0", help="Batch size for training. If batch_size ==0, then input all data for training at once"
    )
    parser.add_argument(
        "--learning_rate", type = float, default = 0.001, help = "the constants for dropout_rate"
    )
    parser.add_argument(
        "--stop_type", type = int, default = 0, help = "stoping type"
    )
    parser.add_argument(
        "--divide_rate", type = float, default = 0.2, help = "divide rate"
    )
    parser.add_argument(
        "--count_lim", type = int, default = 5, help = "specify we stop"
    )
    parser.add_argument(
        "--cov", type = int, default = 0, help = "if cov exists, then set cov as 1"
    )

    args = parser.parse_args()
    return args

act_fn_by_name = {"tanh": nn.Tanh, "relu": nn.ReLU, "identity": nn.Identity, "leakyrelu": nn.LeakyReLU}

args = parse_args()
directory = args.dir
seed, permtime, GPU_NUM, batch_size, learning_rate = args.seed,  args.perm, args.gpu, args.batch_size, args.learning_rate
activation, losstype, reg_type = args.activation, args.loss, args.reg_type
penalty_path_disease,penalty_bio_path, leakyrelu_const, dropout_rate = args.reg_const_pathway_disease, args.reg_const_bio_pathway, args.leakyrelu_const, args.dropout_rate
experiment_name = args.experiment_name
stop_type = args.stop_type
divide_rate = args.divide_rate
count_lim = args.count_lim
cov_exist = args.cov


#stop_type1 = loss with all
#stop_type2 = AUC with all
#stop_type3 = loss with train test divide
#stop_type4 = AUC with train test divide
#stop_type5 = epoch fix with all

os.chdir(directory)
layerinfo = pd.read_csv("layerinfo.csv")
node_num = layerinfo["node_num"].tolist()
layer_num = layerinfo["layer_num"].tolist()

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
 
createFolder(directory + '/' + experiment_name)
createFolder(directory + '/' + experiment_name+ '/tmp')




if activation == "leakyrelu":
    act_fn = act_fn_by_name[activation](leakyrelu_const)
elif activation in act_fn_by_name:
    act_fn = act_fn_by_name[activation]()
else:
    raise NotImplementedError("Not predefined activation function")

dropout_fn = nn.Dropout(dropout_rate)

def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.fill_(0.01)


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

class pathwayblock(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        if hidden_dim == 0:
            self.block = nn.Sequential(
                nn.Linear(input_dim, 1, bias = False),
                act_fn,
                dropout_fn
            )
        else: #layer_num = btw pathway and biological factor number. if layer_num ==0, then hidden num should be zero for setting. we do not allow =0
            modules = []
            modules.append(nn.Linear(input_dim, hidden_dim, bias = False))
            modules.append(act_fn)
            modules.append(dropout_fn)
            for i in range(layer_num-1):
                modules.append(nn.Linear(hidden_dim, hidden_dim, bias = False))
                modules.append(act_fn)
                modules.append(dropout_fn)
            modules.append(nn.Linear(hidden_dim, 1, bias = False))
            modules.append(act_fn)
            modules.append(dropout_fn)
            self.block = nn.Sequential(*modules)
        self.block.apply(init_weights)

    def forward(self, x):
        return self.block(x)



class DeepHisCoM(nn.Module):
    def __init__(self,  nvar, width, layer, covariate, device):
        super(DeepHisCoM, self).__init__()
        self.nvar = nvar
        self.width = width
        self.layer = layer 
        self.pathway_nn = nn.ModuleList([pathwayblock(nvar[i], width[i], layer[i]) for i in range(len(self.nvar))])
        self.bn_path = nn.BatchNorm1d(len(nvar))
        self.dropout_path = dropout_fn
        self.covariate = covariate
        self.fc_path_disease=nn.Linear(len(nvar) +covariate ,1)
        self.fc_path_disease.weight.data.fill_(0)
        self.fc_path_disease.bias.data.fill_(0.001)
        self.device = device

    def forward(self, x):
        kk=0
        nvarlist = list()
        nvarlist.append(kk)
        for i in range(len(self.nvar)):
            k=kk
            kk=kk+self.nvar[i]
            nvarlist.append(kk)
        nvarlist.append(kk + self.covariate)
        pathway_layer = torch.cat([self.pathway_nn[i](x[:,nvarlist[i]:nvarlist[i+1]]) for i in range(len(self.nvar))],1)
        pathway_layer = self.bn_path(pathway_layer)
        pathway_layer = pathway_layer/(torch.norm(pathway_layer,2))
        x = torch.cat([pathway_layer, x[:, nvarlist[len(self.nvar)]:nvarlist[len(self.nvar) + 1]]], 1)
        x = self.dropout_path(x)
        x = self.fc_path_disease(x)
        x = torch.sigmoid(x)
        return(x)


is_cuda = torch.cuda.is_available()                                                                 
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
if is_cuda:
    torch.cuda.set_device(device)                                       # change allocation of current GPU
dtype = torch.FloatTensor

'''
class_weight = torch.FloatTensor([1,1])                                             # if unbalanced data, we can change it for BCELoss
'''

train_base = pd.read_csv("train_data.csv")

for permutation in range(0, permtime):
    save = []
    print("start  ", permutation )
    torch.manual_seed(permutation); random.seed(permutation); np.random.seed(permutation)              # for permutation random seed
    train = train_base
    if permutation != 0:
        train['phenotype'] = np.random.permutation(train['phenotype'])                              # permute y-data
    phenotype = train['phenotype']
    train = (train-train.mean())/train.std()
    train['phenotype'] = phenotype



    train = torch.from_numpy(train.values).type(dtype)
    train_varlen = train.shape[1];    train_datanum = train.shape[0]
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    data_train = CustomDataset(train, train_varlen)

    annot = pd.read_csv("annot.csv")     # annotation data
    groupunique = list(OrderedDict.fromkeys(annot["group"]))   # nvar setting 
    grouplen = len(groupunique)
    nvar = []
    if cov_exist == 1:
        cov = pd.read_csv("cov.csv")     # cov data
        cov_num = len(cov["x"])
    else:
        cov_num = 0



    for i in range(0,grouplen):  
        nvar.append(sum(annot["group"]==groupunique[i]))
    
    if 4<=stop_type:
        TEST_SIZE = divide_rate
        train_indices, test_indices, _, _ = train_test_split(range(len(data_train)), data_train.y_data, stratify = data_train.y_data, 
                                                        test_size = TEST_SIZE)
        print(train_indices)

        train_split = Subset(data_train, train_indices)
        test_split = Subset(data_train, test_indices)
        if batch_size == 0:
            data_train = DataLoader(train_split, batch_size = len(train_split), shuffle = True)
            data_test = DataLoader(test_split, batch_size = len(test_split))
        else:
            data_train = DataLoader(train_split, batch_size = batch_size, shuffle = True)
            data_test = DataLoader(test_split, batch_size = batch_size)
    else:
        if batch_size == 0:
            data_train = DataLoader(data_train, batch_size = len(data_train), shuffle = True)
        else:
            data_train = DataLoader(data_train, batch_size = batch_size, shuffle = True)

    model_DeepHisCoM = DeepHisCoM(nvar, node_num, layer_num, cov_num, device)
    if is_cuda:                                                                         
        model_DeepHisCoM.to('cuda')
    optimizer_DeepHisCoM = torch.optim.Adam([{'params': model_DeepHisCoM.parameters()}]
                                                , lr=learning_rate)                     # can change optimizer

    if losstype == 'bceloss':
        criterion = nn.BCELoss()
    elif losstype == 'mseloss':
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError("Not predefined loss function")
        
    count = 0

    for epoch in range(0,10000):
        save_tmp = []
        for batch_idx, samples in enumerate(data_train):
            model_DeepHisCoM.train()
            optimizer_DeepHisCoM.zero_grad()
            x_train, y_train = samples 
            if is_cuda:                                                             # If we use GPU.
                x_train = x_train.cuda()
                y_train = y_train.cuda()
            output_DeepHisCoM = torch.squeeze(model_DeepHisCoM(x_train))
            y_train = torch.squeeze(y_train)
            loss_DeepHisCoM = (criterion(output_DeepHisCoM, y_train)).type(dtype)
            if penalty_path_disease != 0 :
                for param in model_DeepHisCoM.fc_path_disease.parameters():
                    if reg_type == 'l1':
                        loss_DeepHisCoM = loss_DeepHisCoM + penalty_path_disease * torch.norm(param,1)
                    elif reg_type == 'l2':
                        loss_DeepHisCoM = loss_DeepHisCoM + penalty_path_disease * torch.norm(param,2) * torch.norm(param,2)
                    else:
                        raise NotImplementedError("Not predefined reg function")
            if penalty_bio_path != 0:                    
                for param in model_DeepHisCoM.pathway_nn.parameters():
                    if reg_type == 'l1':
                        loss_DeepHisCoM = loss_DeepHisCoM + penalty_bio_path * torch.norm(param,1)
                    elif reg_type == 'l2':
                        loss_DeepHisCoM = loss_DeepHisCoM + penalty_bio_path * torch.norm(param,2) * torch.norm(param,2)                            
                    else:
                        raise NotImplementedError("Not predefined reg function")
            
            loss_DeepHisCoM.backward()
            optimizer_DeepHisCoM.step()
            if stop_type == 1:
                save_tmp.append(-loss_DeepHisCoM.item())
            if stop_type == 2:
                if torch.sum(torch.isnan(output_DeepHisCoM))==0:
                    AUC_test =  roc_auc_score(y_test.cpu().detach().numpy(), output_DeepHisCoM.cpu().detach().numpy())
                else:
                    AUC_test = 0        
                save_tmp.append(AUC_test)
            if stop_type == 5:
                param_save = np.array(list(model_DeepHisCoM.fc_path_disease.parameters())[0].tolist()[0])

        if 3 <= stop_type <= 4:
            for batch_idx, samples in enumerate(data_test):
                model_DeepHisCoM.eval()
                x_test, y_test = samples
                if is_cuda:                                                             # If we use GPU.
                    x_test = x_test.cuda()
                    y_test = y_test.cuda()
                output_DeepHisCoM = torch.squeeze(model_DeepHisCoM(x_test))
                y_test = torch.squeeze(y_test)

                if stop_type == 3:
                    loss_DeepHisCoM_test = (criterion(output_DeepHisCoM, y_train)).type(dtype).item()
                    save_tmp.append(-loss_DeepHisCoM_test)

                if stop_type == 4:
                    if torch.sum(torch.isnan(output_DeepHisCoM))==0:
                        AUC_test =  roc_auc_score(y_test.cpu().detach().numpy(), output_DeepHisCoM.cpu().detach().numpy())
                    else:
                        AUC_test = 0        
                    save_tmp.append(AUC_test)
                    print(AUC_test)

        if stop_type in [1,2,3,4]:
            save_avg = sum(save_tmp)/len(save_tmp)
            save.append(save_avg)
            if max(save) == save_avg:
                count = 0
                param_save = np.array(list(model_DeepHisCoM.fc_path_disease.parameters())[0].tolist()[0])
            else:
                count = count + 1
            

            if(count > count_lim):
                break
        elif stop_type == 5:
            if epoch > count_lim:
                break
            
    param = param_save
    np.savetxt(experiment_name + "/tmp/param"+ str(permutation) + ".txt" ,param)



#if __name__ == '__main__':
#    set_start_method('spawn')
#    pool = torch.multiprocessing.Pool(processes = multiprocess_num)
#    pool.map(experiment, range(0,total_permutation_num))
#    pool.close()
#    pool.join()
