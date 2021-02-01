import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from collections import OrderedDict
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')
import pandas as pd
import multiprocessing
import sys, os

#one can also use it to gpu. then you should change some code.

class CustomDataset(Dataset): 
  def __init__(self, DATA, inputlength):
    self.x_data = DATA[:,0:(inputlength-1)]
    self.y_data = DATA[:,(inputlength-1)].reshape(-1,1)
  def __len__(self): 
    return len(self.x_data)
  def __getitem__(self, idx): 
    x = torch.FloatTensor(self.x_data[idx])
    y = torch.FloatTensor(self.y_data[idx])
    return x, y

  
class DeepHisCoM(nn.Module):
  def __init__(self,  nvar, node_num):
    super(DeepHisCoM, self).__init__()
    self.nvar = nvar
    self.fc1 = nn.ModuleList([nn.Linear(nvar[i],node_num, bias = False) for i in range(len(self.nvar))])    #fc1, fc4 make pathway-biological factor connections
    self.fc4 = nn.ModuleList([nn.Linear(node_num, 1, bias = False) for i in range(len(self.nvar))])
    self.fc1_bn = nn.BatchNorm1d(len(nvar))
    self.fc2=nn.Linear(len(nvar),1)  #fc2 make pathway-phenotype connection
  def forward(self, x):
    kk=0
    s=list()
    for i, l in enumerate(self.fc1):
      if i==0:
        k=kk
        kk=kk+self.nvar[i]
        t= x[:,k:kk]
        s.append(l(t))
      else:
        k=kk
        kk=kk+self.nvar[i]
        t= x[:,k:kk]
        s.append(F.leaky_relu(l(t),0.2)) 
    for i, l in enumerate(self.fc4):
      if i==0:
        x=l(s[i])
      else:
        x=torch.cat((x,F.leaky_relu(self.fc4[i](s[i]),0.2)),1)
    x=self.fc1_bn(x)
    x = x/(torch.norm(x,2)+0.00000001)   #Normalization (Hwang, 2009)
    x=self.fc2(x)
    return(torch.sigmoid(x))
#can change activation function or layer number or node number (now leaky relu is activation function)


best_epoch = 1          #if you use gradient descent(in here we use L-BFGS), change epoch as hyperparameter
best_learning_rate = 0.1        # for learning rate. you can use only gradient descent also.
bestpenalty_path_phenotype = 0.0001     #pathway between phenotype and pathway
bestpenalty_bf_path = 0    #penalty between biological factor and pathway
best_node_num = 4   #node number 
#selected by cross validation. 
permnum = 1000  #permutation time


for repp in range(0,permnum):
  print(repp)
  train = pd.read_csv("train.csv")  #permutationed data
  Pep = pd.read_csv("annot.csv")  #data loading
  groupunique = list(OrderedDict.fromkeys(Pep["group"]))
  grouplen = len(groupunique)
  nvar = []
  for i in range(0,grouplen):  
    nvar.append(sum(Pep["group"]==groupunique[i]))

  node_num = best_node_num
  learning_rate = best_learning_rate
  penalty_bf_path = bestpenalty_bf_path
  penalty_path_phenotype = bestpenalty_path_phenotype #cross validation result loading
  
  train = torch.from_numpy(train.values).float()
  data_train = CustomDataset(train, (train.shape[1]))
  input_size = train.shape[1]

  pathway_size = grouplen
  num_classes = 2
  data_train = DataLoader(data_train, batch_size = train.shape[0], shuffle = True)

  model_DeepHisCoM = DeepHisCoM(nvar, node_num)
  optimizer_DeepHisCoM = torch.optim.LBFGS([{'params': model_DeepHisCoM.parameters()}], lr=learning_rate, max_iter = 50000, max_eval = None, tolerance_change= 1e-07, history_size = 100, line_search_fn = "strong_wolfe") #L-BFGS optimizer
  class_weight = torch.FloatTensor([1,1])
  criterion = nn.BCELoss()
  for epoch in range(0,1):
    for batch_idx, samples in enumerate(data_train):
      def closure():
        optimizer_DeepHisCoM.zero_grad()
        x_train, y_train = samples
        output_DeepHisCoM = model_DeepHisCoM(x_train)
        output_DeepHisCoM = torch.squeeze(output_DeepHisCoM)
        y_train = torch.squeeze(y_train)
        loss_DeepHisCoM = criterion(output_DeepHisCoM, y_train)
        for param in model_DeepHisCoM.fc2.parameters():
          loss_DeepHisCoM = loss_DeepHisCoM + penalty_path_phenotype * torch.norm(param,2) * torch.norm(param,2)
        for x in model_DeepHisCoM.fc4:
          for param in x.parameters():
            loss_DeepHisCoM = loss_DeepHisCoM + penalty_bf_path * torch.norm(param,2)* torch.norm(param,2)
        for x in model_DeepHisCoM.fc1:
          for param in x.parameters():
            loss_DeepHisCoM = loss_DeepHisCoM + penalty_bf_path * torch.norm(param,2)* torch.norm(param,2)     
        loss_DeepHisCoM=  loss_DeepHisCoM.type(torch.FloatTensor)
        loss_DeepHisCoM.backward()
        return(loss_DeepHisCoM)
      loss = optimizer_DeepHisCoM.step(closure)
  param = list(model_DeepHisCoM.fc2.parameters())[0]
  param = np.array(param.tolist()[0])
  param_file = "np.savetxt('param" + str(repp) + ".txt',param)"
  eval(param_file)
