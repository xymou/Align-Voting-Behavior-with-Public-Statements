import json
import os
import re
import pickle
import random
import math
import argparse
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import pdist
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import nltk
from nltk.corpus import stopwords
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings 
warnings.filterwarnings("ignore") 
cuda0=torch.device('cuda:{}'.format(1)) #assign cuda device, keep the same as that in train.py

def save_obj(obj, name):
    with open('obj'+ name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name,path=None):
    if path is None:
        with open('obj' + name + '.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        with open(path+'obj' + name + '.pkl', 'rb') as f:
            return pickle.load(f)        

#shuffle dataset
def random_shuffle(node,relation,label, tag_label):
    lines=[]
    with open(node) as fp:
        lines=fp.readlines()
    fp.close()
    random.shuffle(lines)
    with open(node,'w') as fp:
        for line in lines:
            fp.write(line)
    fp.close()
    lines=[]
    with open(relation) as fp:
        lines=fp.readlines()
    fp.close()
    random.shuffle(lines)
    with open(relation,'w') as fp:
        for line in lines:
            fp.write(line)
    fp.close()    
    lines=[]
    with open(label) as fp:
        lines=fp.readlines()
    fp.close()
    random.shuffle(lines)
    with open(label,'w') as fp:
        for line in lines:
            fp.write(line)
    fp.close() 
    lines=[]
    with open(tag_label) as fp:
        lines=fp.readlines()
    fp.close()
    random.shuffle(lines)
    with open(tag_label,'w') as fp:
        for line in lines:
            fp.write(line)
    fp.close()     
    
def shuffle(path):
    lines=[]
    with open(path) as fp:
        lines=fp.readlines()
    fp.close()
    random.shuffle(lines)
    with open(path,'w') as fp:
        for line in lines:
            fp.write(line)
    fp.close()

#split validation set from train set
def prepare(label_train):
    mems, legs, labels=[],[],[]
    with open(label_train) as fp:
        for i, line in enumerate(fp):
            info=line.strip().split()
            node1=info[0]
            node2=info[1]
            label=info[2]
            mems.append(node1)
            legs.append(node2)
            labels.append(label)  
    fp.close()
    leg_ids=list(set(legs))
    random.shuffle(leg_ids)
    dev_ids=leg_ids[:len(leg_ids)//5]
    train_ids=leg_ids[(len(leg_ids)//5):]
    label_dev='label_dev_tmp.txt'
    label_train1='label_train_tmp.txt'
    with open(label_dev,'w') as f:
        for i in range(len(legs)):
            if legs[i] in dev_ids:
                f.write(mems[i]+' '+legs[i]+' '+labels[i])
                f.write('\n')
    f.close()
    with open(label_train1,'w') as f:
        for i in range(len(legs)):
            if legs[i] in train_ids:
                f.write(mems[i]+' '+legs[i]+' '+labels[i])
                f.write('\n')          
    return label_train1,label_dev    



def normalize(A, func='median', eliminate=False):
    #normalize the adjacency matrix
    if eliminate:
        flatten_adj = A.flatten()
        flatten_adj = np.ma.masked_equal(flatten_adj,0.0).compressed()
        median_edge = np.median(flatten_adj) if func=='median' else np.mean(flatten_adj)
        A[A<=median_edge] = 0    
    for i in range(len(A)):
        if sum(A[i,:]):
            A[i,:]=A[i,:]/sum(A[i,:])
    return A


#dataloader
class mydata(object):
    def __init__(self, num_nodes=6166, num_feats=768):
        self.feats=np.zeros((num_nodes,num_feats))
        self.num_feats=num_feats
        self.batches=[]
        self.mem_nodes, self.leg_nodes, self.tag_nodes=[],[],[]
        self.nodes=[] 
    
    def prepare(self,node_path, relation_path):
        node2type=load_obj("node2type",'./data/')
        mem_lookup=[]
        leg_lookup=[]
        tag_lookup=[]
        with open(node_path) as fp:
            for i, line in enumerate(fp):
                info=line.strip().split()
                idx=int(info[0])   
                self.nodes.append(int(info[0]))
                if node2type[idx]=='mem':
                    mem_lookup.append(idx)
                    self.feats[idx,:]=list(map(float,info[1:]))+[0]*(768-64)
                    self.mem_nodes.append(int(info[0]))
                elif node2type[idx]=='leg':
                    leg_lookup.append(idx)
                    self.feats[idx,:]=list(map(float,info[1:]))
                    self.leg_nodes.append(int(info[0]))
                else:
                    tag_lookup.append(idx)
                    self.feats[idx,:]=list(map(float,info[1:]))
                    self.tag_nodes.append(int(info[0]))
        self.mem_lookup={m:mem_lookup.index(m) for m in mem_lookup}
        self.leg_lookup={l:leg_lookup.index(l) for l in leg_lookup}
        self.tag_lookup={t:tag_lookup.index(t) for t in tag_lookup} 
        mem_feats=np.zeros((len(self.mem_lookup), 64))
        leg_feats=np.zeros((len(self.leg_lookup), 768))
        tag_feats=np.zeros((len(self.tag_lookup), 768))
        for m in mem_lookup:
            mem_feats[self.mem_lookup[m],:]=self.feats[m,:64]
        for l in leg_lookup:
            leg_feats[self.leg_lookup[l],:]=self.feats[l,:768]
        for t in tag_lookup:
            tag_feats[self.tag_lookup[t],:]=self.feats[t,:768]
            
        #get adj matrix from relation files
        mem2mem_matrix=np.zeros((len(self.mem_lookup),len(self.mem_lookup)),dtype='float32')
        mem2leg_matrix=np.zeros((len(self.mem_lookup),len(self.leg_lookup)),dtype='float32')
        mem2tag_matrix=np.zeros((len(self.mem_lookup),len(self.tag_lookup)),dtype='float32')
        leg2leg_matrix=np.zeros((len(self.leg_lookup),len(self.leg_lookup)),dtype='float32')
        tag2tag_matrix=np.zeros((len(self.tag_lookup),len(self.tag_lookup)),dtype='float32')
        leg2tag_matrix=np.zeros((len(self.leg_lookup),len(self.tag_lookup)),dtype='float32')
                                
        with open(relation_path) as fp:
            for i, line in enumerate(fp):
                info=line.strip().split()
                node1=int(info[0])
                node2=int(info[1])  
                weight=float(info[2])
                r_type=info[3]
                if r_type=='mem2mem':
                    mem2mem_matrix[self.mem_lookup[node1]][self.mem_lookup[node2]]=weight
                elif r_type=='mem2leg':
                    mem2leg_matrix[self.mem_lookup[node1]][self.leg_lookup[node2]]=weight
                elif r_type=='mem2tag':
                    mem2tag_matrix[self.mem_lookup[node1]][self.tag_lookup[node2]]=weight
                elif r_type=='leg2leg':
                    leg2leg_matrix[self.leg_lookup[node1]][self.leg_lookup[node2]]=weight
                elif r_type=='tag2tag':
                    tag2tag_matrix[self.tag_lookup[node1]][self.tag_lookup[node2]]=weight
                elif r_type=='leg2tag':
                    leg2tag_matrix[self.leg_lookup[node1]][self.tag_lookup[node2]]=weight

        mem2mem_matrix=normalize(mem2mem_matrix,'mean',True)
        leg2mem_matrix=normalize(mem2leg_matrix.copy().T,'mean',False)
        mem2leg_matrix=normalize(mem2leg_matrix,'mean',False)
        tag2mem_matrix=normalize(mem2tag_matrix.copy().T,'mean',True)
        mem2tag_matrix=normalize(mem2tag_matrix,'mean',True)
        leg2leg_matrix=normalize(leg2leg_matrix,'mean',True)
        tag2leg_matrix=normalize(leg2tag_matrix.copy().T,'mean',True)
        leg2tag_matrix=normalize(leg2tag_matrix,'mean',True)
        tag2tag_matrix=normalize(tag2tag_matrix,'mean',True)                                   
        
        #construct adj matrix for social network to compute following loss
        self.adj_lists=[[]]*6616
        self.weight_lists=[[]]*6616
        lookup_mem={k:v for v,k in self.mem_lookup.items()}
        lookup_leg={k:v for v,k in self.leg_lookup.items()}
        lookup_tag={k:v for v,k in self.tag_lookup.items()} 
        follow_dict=load_obj('follow_dict','./data/')      
        self.lookup_mem=lookup_mem
        for key in follow_dict:
            if key not in self.mem_lookup:
                continue
            tmp=[t for t in follow_dict[key] if t in self.mem_lookup]
            self.adj_lists[key]=tmp
            self.weight_lists[key]=[1]*len(tmp)               
            
        return  torch.tensor(mem_feats, dtype=torch.float32).to(cuda0),torch.tensor(leg_feats, dtype=torch.float32).to(cuda0), torch.tensor(tag_feats, dtype=torch.float32).to(cuda0), torch.tensor(mem2mem_matrix, dtype=torch.float32).to(cuda0), torch.tensor(mem2leg_matrix, dtype=torch.float32).to(cuda0), \
                    torch.tensor(mem2tag_matrix, dtype=torch.float32).to(cuda0), torch.tensor(leg2leg_matrix, dtype=torch.float32).to(cuda0), torch.tensor(leg2tag_matrix, dtype=torch.float32).to(cuda0), torch.tensor(tag2tag_matrix, dtype=torch.float32).to(cuda0),\
                    torch.tensor(leg2mem_matrix, dtype=torch.float32).to(cuda0),torch.tensor(tag2mem_matrix, dtype=torch.float32).to(cuda0),torch.tensor(tag2leg_matrix, dtype=torch.float32).to(cuda0),
    
    def batcher(self,batch_size,label_path,tag_label_path):
        mems, legs, labels=[],[],[]
        mems2,tags, labels2=[],[],[]
        with open(label_path) as fp:
            for i, line in enumerate(fp):
                info=line.strip().split()
                node1=self.mem_lookup[int(info[0])]
                node2=self.leg_lookup[int(info[1])]
                label=info[2]
                if label=='2':
                    continue
                mems.append(node1)
                legs.append(node2)
                labels.append(label)
                                          
        with open(tag_label_path) as fp:
            for i, line in enumerate(fp):
                info=line.strip().split()
                node1=self.mem_lookup[int(info[0])]
                node2=self.tag_lookup[int(info[1])]
                label=info[2]
                mems2.append(node1)
                tags.append(node2)
                labels2.append(label) 
       
        #batch size of hashtag label can be inferred from ratio of vote labels and hashtag labels
        batch_size2=int(batch_size*len(labels2)/len(labels))
        if len(legs) % batch_size==0:
            batch_num=int(len(legs)/batch_size)
        else:
            batch_num=math.floor(len(legs)/batch_size)+1
        i=0
        for i in range(batch_num-1):
            item={}
            item['leg']=legs[i*batch_size:(i+1)*batch_size]
            item['mem']=mems[i*batch_size:(i+1)*batch_size]
            item['label']=labels[i*batch_size:(i+1)*batch_size]
            item['tag']=tags[i*batch_size2:(i+1)*batch_size2]
            item['mem2']=mems2[i*batch_size2:(i+1)*batch_size2]
            item['label2']=labels2[i*batch_size2:(i+1)*batch_size2]            
            self.batches.append(item)
        item={}
        item['leg']=legs[i*batch_size:]
        item['mem']=mems[i*batch_size:]
        item['label']=labels[i*batch_size:] 
        item['tag']=tags[i*batch_size2:]
        item['mem2']=mems2[i*batch_size2:]
        item['label2']=labels2[i*batch_size2:]         
        self.batches.append(item)
        self.legs, self.mems, self.labels, self.tags, self.mems2, self.labels2 =legs, mems, labels, tags, mems2, labels2

def _init_weights(module):
    if isinstance(module,nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)

class RGCN(nn.Module):
    def __init__(self,dim_in,hidden_size,dim_out):
        super(RGCN,self).__init__()
        self.fc_m1=nn.Linear(64,128,bias=False)
        self.fc_l1=nn.Linear(768,128,bias=False)
        self.fc_t1=nn.Linear(768,128,bias=False)
        self.fc_m2=nn.Linear(128,64,bias=False)
        self.fc_l2=nn.Linear(128,64,bias=False)
        self.fc_t2=nn.Linear(128,64,bias=False)          
        self.fc_m2m=nn.Linear(64,128,bias=False)
        self.fc_m2l=nn.Linear(768,128,bias=False)
        self.fc_m2t=nn.Linear(768,128,bias=False)
        self.fc_l2l=nn.Linear(768,128,bias=False) 
        self.fc_l2t=nn.Linear(768,128,bias=False) 
        self.fc_t2t=nn.Linear(768,128,bias=False)
        self.fc_l2m=nn.Linear(64,128,bias=False)
        self.fc_t2m=nn.Linear(64,128,bias=False)
        self.fc_t2l=nn.Linear(768,128,bias=False)
        self.fc_m2m_2=nn.Linear(128,64,bias=False)
        self.fc_m2l_2=nn.Linear(128,64,bias=False)
        self.fc_m2t_2=nn.Linear(128,64,bias=False)
        self.fc_l2l_2=nn.Linear(128,64,bias=False)
        self.fc_l2t_2=nn.Linear(128,64,bias=False)
        self.fc_t2t_2=nn.Linear(128,64,bias=False)
        self.fc_l2m_2=nn.Linear(128,64,bias=False)
        self.fc_t2m_2=nn.Linear(128,64,bias=False)
        self.fc_t2l_2=nn.Linear(128,64,bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(768)
        self.bn3 = nn.BatchNorm1d(768)
        self.dropout=nn.Dropout(0.5)
        self.apply(_init_weights)
    
    def forward(self,X, Y, Z, mem2mem, mem2leg, mem2tag, leg2leg, leg2tag, tag2tag, leg2mem, tag2mem, tag2leg): 
        X= self.bn1(X)
        Y= self.bn2(Y)
        Z= self.bn3(Z)
        X1=F.relu(self.fc_m1(X)+self.fc_m2m(mem2mem.mm(X))+self.fc_m2l(mem2leg.mm(Y))+self.fc_m2t(mem2tag.mm(Z)))
        Y1=F.relu(self.fc_l1(Y)+self.fc_l2l(leg2leg.mm(Y))+self.fc_l2m(leg2mem.mm(X))+self.fc_l2t(leg2tag.mm(Z)))
        Z1=F.relu(self.fc_t1(Z)+self.fc_t2t(tag2tag.mm(Z))+self.fc_t2m(tag2mem.mm(X))+self.fc_t2l(tag2leg.mm(Y)))
        X1,Y1,Z1=self.dropout(X1),self.dropout(Y1),self.dropout(Z1)
  
        X_output=self.fc_m2(X1)+self.fc_m2m_2(mem2mem.mm(X1))+self.fc_m2l_2(mem2leg.mm(Y1))+self.fc_m2t_2(mem2tag.mm(Z1))
        Y_output=self.fc_l2(Y1)+self.fc_l2l_2(leg2leg.mm(Y1))+self.fc_l2m_2(leg2mem.mm(X1))+self.fc_l2t_2(leg2tag.mm(Z1))
        Z_output=self.fc_t2(Z1)+self.fc_t2t_2(tag2tag.mm(Z1))+self.fc_t2m_2(tag2mem.mm(X1))+self.fc_t2l_2(tag2leg.mm(Y1))
        X_output, Y_output, Z_output=self.dropout(X_output),self.dropout(Y_output),self.dropout(Z_output)
        return X_output, Y_output, Z_output


class HGCN(nn.Module):
    def __init__(self,dim_in,hidden_size,dim_out, label_num1,label_num2):
        super(HGCN, self).__init__()
        self.rgcn=RGCN(dim_in,hidden_size,dim_out)
        self.fc1=nn.Linear(dim_out*2,label_num1,bias=True)
        self.fc2=nn.Linear(dim_out*2,label_num2,bias=True)
        self.apply(_init_weights)
    def forward(self,X, Y, Z, mem2mem, mem2leg, mem2tag, leg2leg, leg2tag, tag2tag,leg2mem, tag2mem, tag2leg, mems, legs, mems2, tags):
        X_output, Y_output, Z_output=self.rgcn(X, Y, Z, mem2mem, mem2leg, mem2tag, leg2leg, leg2tag, tag2tag,leg2mem, tag2mem, tag2leg)
        leg_embs=Y_output[legs]
        mem_embs=X_output[mems]
        out1=torch.mul(leg_embs,mem_embs)
        out2=leg_embs-mem_embs
        out=torch.cat([out1,out2],dim=1)
        logits1=self.fc1(out)
        
        tag_embs=Z_output[tags]
        mem2_embs=X_output[mems2]
        out1=torch.mul(tag_embs,mem2_embs)
        out2=tag_embs-mem2_embs
        out=torch.cat([out1,out2],dim=1)
        logits2=self.fc2(out)
        return logits1,logits2
