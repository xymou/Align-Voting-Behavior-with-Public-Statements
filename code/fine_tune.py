# -*- coding: utf-8 -*-
"""
@author: xymou
mode2: fine tune BERT and Member embedding modules
This mode can be very slow since before every time you train, you need to  re-encode the nodes. 
But you can still use relations and labels stored in the txt files to save time.
"""

from pytorch_pretrained_bert import BertModel, BertTokenizer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import json
import os
import re
import numpy as np
import pickle
from tqdm import tqdm
from scipy.spatial.distance import pdist
from scipy import special
import nltk
from nltk.corpus import stopwords
from collections import Counter, OrderedDict
import random
import math
import argparse
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings 
warnings.filterwarnings("ignore") 
from model.model import *
from model.walker import *
from prepare import *

stws = stopwords.words('english')
stws+=['[URL]','[MENTION]','[PIC]','[url]','[mention]','[pic]','w/','$','rt','via','--','&','[url]…','-']
stws+=[str(i) for i in list(range(10))]
stws+= [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '/', '-',
                        '<', '>','’']
tokenizer = BertTokenizer.from_pretrained('/remote-home/xymou/bert/bert-base-uncased-vocab.txt')

def padding(idx_lst):
    seq_len=[idx.size(1) for idx in idx_lst]
    max_len=max(seq_len)
    idx_list=[]
    for idx in idx_lst:
        idx_list.append(torch.cat([idx,torch.zeros((1,max_len-idx.size(1)))],dim=1))
    return idx_list,seq_len

def save_obj(obj, name):
    with open('obj' + name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name,path=None):
    if path is None:
        with open('obj' + name + '.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        with open(path+'obj' + name + '.pkl', 'rb') as f:
            return pickle.load(f) 


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


def normalize(A, func='median', eliminate=False):
    if eliminate:
        flatten_adj = A.flatten()
        flatten_adj = np.ma.masked_equal(flatten_adj,0.0).compressed()
        median_edge = np.median(flatten_adj) if func=='median' else np.mean(flatten_adj)
        A[A<=median_edge] = 0    
    for i in range(len(A)):
        if sum(A[i,:]):
            A[i,:]=A[i,:]/sum(A[i,:])
    return A

class mydata(object):
    def __init__(self, num_nodes=6166):
        self.batches=[]
        self.mem_nodes, self.leg_nodes, self.tag_nodes=[],[],[]
        self.nodes=[] 
    
    def prepare(self,node_path, relation_path):     
        node2type=load_obj("node2type",'./data/')
        mem_lookup=[]
        leg_lookup=[]
        tag_lookup=[]

        with open(node_path) as fp:
            for i,line in enumerate(fp):
                idx=int(line.strip().split()[0])
                self.nodes.append(idx)
                if node2type[idx]=='mem':
                    mem_lookup.append(idx)
                    self.mem_nodes.append(idx)
                elif node2type[idx]=='leg':
                    leg_lookup.append(idx)
                    self.leg_nodes.append(idx)
                else:
                    tag_lookup.append(idx)
                    self.tag_nodes.append(idx)     
        
        self.mem_lookup={m:mem_lookup.index(m) for m in mem_lookup}
        self.leg_lookup={l:leg_lookup.index(l) for l in leg_lookup}
        self.tag_lookup={t:tag_lookup.index(t) for t in tag_lookup} 
        mem_feats=[0 for _ in range(len(self.mem_lookup))]
        leg_feats=[0 for _ in range(len(self.leg_lookup))]
        tag_feats=[0 for _ in range(len(self.tag_lookup))]
        for m in mem_lookup:
            mem_feats[self.mem_lookup[m]]=m
        for l in leg_lookup:
            leg_feats[self.leg_lookup[l]]=l
        for t in tag_lookup:
            tag_feats[self.tag_lookup[t]]=t
            
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
        
        #construct adj matrix for following proximity loss
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
            
        return mem_feats, leg_feats, tag_feats, torch.tensor(mem2mem_matrix, dtype=torch.float32).to(cuda0), torch.tensor(mem2leg_matrix, dtype=torch.float32).to(cuda0), \
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

class mem_encoder(nn.Module):
    def __init__(self,member_size, state_size, party_size):
        super(mem_encoder,self).__init__()
        self.member_embedding=nn.Embedding(member_size,32)
        self.state_embedding=nn.Embedding(state_size,16)
        self.party_embedding=nn.Embedding(party_size,16)
    def encode(self,member,state,party):
        member_embed = self.member_embedding(member)
        party_embed = self.party_embedding(party)
        state_embed = self.state_embedding(state)
        legislator_embem = torch.cat([member_embed, party_embed, state_embed], dim=1)
        return legislator_embem

def mem_process(idx):
    mem2state=load_obj('mem2state','./data/')
    mem2party=load_obj('mem2party','./data/')  
    node2info=load_obj('node2info','./data/')
    member_all=np.array(idx)
    state_size=len(list(set(mem2state.values())))
    party_size=len(list(set(mem2party.values())))
    state_all=np.array([mem2state[node2info[m]] for m in idx])
    party_all=np.array([mem2party[node2info[m]] for m in idx])
    return state_size,party_size,torch.LongTensor(member_all).to(cuda0), torch.LongTensor(state_all).to(cuda0), torch.LongTensor(party_all).to(cuda0)

def leg_process(idx):
    bill2text=load_obj('bill2text','./data/')
    node2info=load_obj('node2info','./data/')
    text=[]
    for b in idx:
        b=bill2text[node2info[b]]
        tokens = tokenizer.tokenize(b)
        ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)]).view(1,-1)
        text.append(ids)
    idx_list,seq_len=padding(text)
    idx_list=torch.LongTensor(torch.cat(idx_list,dim=0).numpy()).to(cuda0)
    return idx_list,seq_len

def tag_process(idx):
    tag_text_lookup=load_obj('tag_text','./data/')
    node2info=load_obj('node2info','./data/')
    text=[]
    texts=tag_text_lookup[node2info[idx]]
    for t in texts:
        tokens = tokenizer.tokenize(t)
        ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)]).view(1,-1)
        text.append(ids)    
    idx_list,seq_len=padding(text)
    idx_list=torch.LongTensor(torch.cat(idx_list,dim=0).numpy()).to(cuda0)
    return idx_list, seq_len

def _init_weights(module):
    if isinstance(module,nn.Linear):
        nn.init.xavier_uniform_(module.weight.data) 
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

class HGCN(nn.Module):
    def __init__(self,dim_in,hidden_size,dim_out, label_num1,label_num2,member_size,state_size,party_size):
        super(HGCN, self).__init__()
        self.mem_encoder=mem_encoder(member_size,state_size,party_size)
        #load bert, and freeze layers
        self.text_encoder=BertModel.from_pretrained('/remote-home/xymou/bert/bert-base-uncased/')
        for param in list(self.text_encoder.parameters())[:-1]:
            param.requires_grad = False  
        self.rgcn=RGCN(dim_in,hidden_size,dim_out)
        self.fc1=nn.Linear(dim_out*2,label_num1,bias=True)
        self.fc2=nn.Linear(dim_out*2,label_num2,bias=True)
        self.apply(_init_weights)
    def forward(self,member_all,state_all,party_all, Y, Z, mem2mem, mem2leg, mem2tag, leg2leg, leg2tag, tag2tag,leg2mem, tag2mem, tag2leg, mems, legs, mems2, tags):
        X=self.mem_encoder.encode(member_all, state_all, party_all)
        idx_list,seq_len=leg_process(Y)
        Y=self.text_encoder(idx_list,output_all_encoded_layers=False)[0][:,0,:]
        Z_emb=[]
        for idx in Z:
            idx_list,seq_len=tag_process(idx)
            emb=self.text_encoder(idx_list,output_all_encoded_layers=False)[0][:,0,:]
            emb=torch.mean(emb,dim=0)
            Z_emb.append(emb.view(1,-1))
        Z_emb=torch.cat(Z_emb,dim=0)
        X_output, Y_output, Z_output=self.rgcn(X, Y, Z_emb, mem2mem, mem2leg, mem2tag, leg2leg, leg2tag, tag2tag,leg2mem, tag2mem, tag2leg)
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

def train_fn(args):
    cs_loss=nn.CrossEntropyLoss()
    random_shuffle(node_train, relation_train, label_train, tag_label_train)
    random_shuffle(node_test, relation_test, label_test, tag_label_test)  
    
    #prepare data
    label_train1,label_val=prepare(label_train)
    train=mydata(6166)
    X, Y, Z, mem2mem_matrix, mem2leg_matrix, mem2tag_matrix, leg2leg_matrix,\
                leg2tag_matrix, tag2tag_matrix,leg2mem_matrix, tag2mem_matrix, tag2leg_matrix=train.prepare(node_train, relation_train)
    train.batcher(128,label_train1,tag_label_train)
        
    with open(label_val) as fp:
        df=fp.readlines()
    fp.close()
    val=mydata(6166)
    X1, Y1, Z1, mem2mem_matrix1, mem2leg_matrix1, mem2tag_matrix1, leg2leg_matrix1,\
                leg2tag_matrix1, tag2tag_matrix1, leg2mem_matrix1, tag2mem_matrix1, tag2leg_matrix1= val.prepare(node_val, relation_val)
    val.batcher(len(df),label_val,tag_label_val)     
    
    with open(label_test) as fp:
        df=fp.readlines()
    fp.close()
    test=mydata(6166)
    X2, Y2, Z2, mem2mem_matrix2, mem2leg_matrix2, mem2tag_matrix2, leg2leg_matrix2,\
                leg2tag_matrix2, tag2tag_matrix2, leg2mem_matrix2, tag2mem_matrix2, tag2leg_matrix2= test.prepare(node_test, relation_test)
    test.batcher(len(df),label_test,tag_label_test) 
    
    #use size of all members(906) to instantiate member_ecnoder, or index out of range may occur
    state_size, party_size, member_all, state_all,party_all=mem_process(X)
    model=HGCN(64,64,64,2,2,906,state_size,party_size).to(cuda0) 
    walker=Walker(train.adj_lists,train.weight_lists,train.mem_nodes,train.mems, args.cache_path)
    #optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)     
    params = [
    {"params": model.mem_encoder.parameters(), "lr": args.lr},   
    {"params": model.text_encoder.parameters(), "lr": args.lr},
    {"params": model.rgcn.parameters(), "lr": args.lr*2},
    {"params": model.fc1.parameters(), "lr": args.lr},   
    {"params": model.fc2.parameters(), "lr": args.lr},
    ]
    optimizer=torch.optim.Adam(params)  
    
    _context_losses, _leg_cs_losses, _tag_cs_losses = [], [], []
    count=0
    loss_avg=0
    best_acc=0
    best_acc2=0
    early_stop_count = 0
    
    #begin training
    for epoch in range(args.epochs):
        print('epoch' + str(epoch))
        random.shuffle(train.nodes)
        
        for item in train.batches:
            model.train()
            count+=1
            optimizer.zero_grad()
            walker.reset()
            leg=list(map(int,item['leg']))
            mem=list(map(int,item['mem']))
            tag=list(map(int,item['tag']))
            mem2=list(map(int,item['mem2']))
            logits1,logits2=model(member_all,state_all,party_all, Y, Z, mem2mem_matrix, mem2leg_matrix, mem2tag_matrix, leg2leg_matrix,\
                        leg2tag_matrix, tag2tag_matrix, leg2mem_matrix, tag2mem_matrix, tag2leg_matrix, mem, leg,mem2,tag)            
            targets1=torch.LongTensor(list(map(int,item['label']))).to(cuda0)
            targets2=torch.LongTensor(list(map(int,item['label2']))).to(cuda0)
            leg_cs_loss=cs_loss(logits1,targets1)
            tag_cs_loss=cs_loss(logits2,targets2)
            
            if args.proximity:
                extended_nodes=np.asarray(list(walker.extend_nodes(train.nodes)))  
                extended_mem_batch=[n for n in extended_nodes if n in train.mem_nodes]
                lookup_mem_batch=[train.mem_lookup[n] for n in extended_mem_batch]
                extended_mem_emb_batch=X[lookup_mem_batch] if len(extended_mem_batch)>0 else None             
                node_embed_batch_list, nodes_batch = [], []
                if extended_mem_emb_batch is not None:
                    node_embed_batch_list.append(extended_mem_emb_batch)
                    nodes_batch.extend(extended_mem_batch)
                node_emb_batch = torch.cat(node_embed_batch_list, dim=0)
                if args.unsup_loss=='normal':
                    context_loss=walker.compute_unsup_loss_normal(node_emb_batch, nodes_batch)
                elif args.unsuo_loss=='margin':
                    context_loss=walker.compute_unsup_loss_margin(node_emb_batch, nodes_batch)
                else:
                    raise ValueError("Unsupported unsup loss {}".format(unsup_loss))   
            else:
                context_loss=None
            total_loss=0
            losses=[context_loss, leg_cs_loss, tag_cs_loss]
            ratio=args.ratios.split()
            ratio=[float(r) for r in ratio]
            for i in range(len(losses)):
                if losses[i] is not None:
                    total_loss+=losses[i]*float(ratio[i])
            loss_avg+=total_loss.item()
            total_loss.backward()
            optimizer.step()
            if context_loss is not None:
                _context_losses.append(float(context_loss.detach().cpu()))
            if leg_cs_loss is not None:
                _leg_cs_losses.append(float(leg_cs_loss.detach().cpu()))
            if tag_cs_loss is not None:
                _tag_cs_losses.append(float(tag_cs_loss.detach().cpu()))            
                
            #evaluate
            if count %50==1:
                model.eval()
                print('Loss of Train set is %.4f' % (loss_avg/200.0))
                loss_avg=0
                with torch.no_grad():
                    #prediction
                    item1=val.batches[0]
                    leg=list(map(int,item1['leg']))
                    mem=list(map(int,item1['mem']))
                    tag=list(map(int,item1['tag']))
                    mem2=list(map(int,item1['mem2']))                    
                    targets1=torch.LongTensor(list(map(int,item1['label']))).to(cuda0)
                    targets2=torch.LongTensor(list(map(int,item1['label2']))).to(cuda0)
                    state_size, party_size, member_all, state_all,party_all=mem_process(X1)
                    logits1,logits2=model(member_all,state_all,party_all, Y1, Z1, mem2mem_matrix1, mem2leg_matrix1, mem2tag_matrix1, leg2leg_matrix1,\
                            leg2tag_matrix1, tag2tag_matrix1, leg2mem_matrix1, tag2mem_matrix1, tag2leg_matrix1, mem, leg, mem2, tag)
                    acc1 = int(torch.sum(torch.argmax(logits1, dim=1) == targets1))/int(targets1.shape[0])
                    acc2 = int(torch.sum(torch.argmax(logits2, dim=1) == targets2))/int(targets2.shape[0])
                    y_pred=(torch.argmax(logits1, dim=1).cpu().numpy()-1)*(-1)
                    y_true=(targets1.cpu().numpy()-1)*(-1)
                    f1=f1_score(y_true,y_pred)
                    macro=f1_score(y_true, y_pred, average='macro')
                    print('Val: acc | F1 | macro F1')    
                    print(acc1,f1,macro)
                    if best_acc<acc1:
                        best_acc=acc1
                        best_model=HGCN(64,64,64,2,2,906,state_size,party_size).to(cuda0)
                        best_model.load_state_dict(copy.deepcopy(model.state_dict()))
                        early_stop_count = 0
                    early_stop_count += 1
                    if early_stop_count >=10:
                        break
                    if best_acc2<acc2:
                        best_acc2=acc2
                    print('HGCN acc:'+str(acc1)+ '\t' + 'epoch:' + str(epoch) +
                        '\t\t' + 'Data: '+'Train:'+str(train_time_begin)+'-'+str(train_time_end)+'\t'+'Test:'+str(test_time_begin)+'-'+
                        str(test_time_end)+'\t'+'Best acc:'+str(best_acc))    
                    print('Hashtag post prediction acc:'+str(acc2)+'\t'+'Best acc:'+str(best_acc2))
    
    best_model.eval()
    with torch.no_grad():
        item1=val.batches[0]
        leg=list(map(int,item1['leg']))
        mem=list(map(int,item1['mem']))
        tag=list(map(int,item1['tag']))
        mem2=list(map(int,item1['mem2']))                    
        targets1=torch.LongTensor(list(map(int,item1['label']))).to(cuda0)
        targets2=torch.LongTensor(list(map(int,item1['label2']))).to(cuda0)
        state_size, party_size, member_all, state_all,party_all=mem_process(X1)
        logits1,logits2=best_model(member_all,state_all,party_all, Y1, Z1, mem2mem_matrix1, mem2leg_matrix1, mem2tag_matrix1, leg2leg_matrix1,\
                leg2tag_matrix1, tag2tag_matrix1, leg2mem_matrix1, tag2mem_matrix1, tag2leg_matrix1, mem, leg, mem2, tag)
        acc1 = int(torch.sum(torch.argmax(logits1, dim=1) == targets1))/int(targets1.shape[0])
        acc2 = int(torch.sum(torch.argmax(logits2, dim=1) == targets2))/int(targets2.shape[0])        
        y_pred=(torch.argmax(logits1, dim=1).cpu().numpy()-1)*(-1)
        y_true=(targets1.cpu().numpy()-1)*(-1)
        f1=f1_score(y_true,y_pred)
        macro=f1_score(y_true, y_pred, average='macro')
        print('Val: acc | F1 | macro F1')
        print(acc1,f1,macro)
    
        item1=test.batches[0]
        leg=list(map(int,item1['leg']))
        mem=list(map(int,item1['mem']))
        mask=[i for i in range(len(mem)) if test.lookup_mem[mem[i]] in train.mem_lookup]
        tag=list(map(int,item1['tag']))
        mem2=list(map(int,item1['mem2']))                    
        targets1=torch.LongTensor(list(map(int,item1['label']))).to(cuda0)
        targets2=torch.LongTensor(list(map(int,item1['label2']))).to(cuda0)
        state_size, party_size, member_all, state_all,party_all=mem_process(X2)
        logits1,logits2=best_model(member_all,state_all,party_all, Y2, Z2, mem2mem_matrix2, mem2leg_matrix2, mem2tag_matrix2, leg2leg_matrix2,\
                leg2tag_matrix2, tag2tag_matrix2, leg2mem_matrix2, tag2mem_matrix2, tag2leg_matrix2, mem, leg, mem2, tag)
        acc1 = int(torch.sum(torch.argmax(logits1, dim=1) == targets1))/int(targets1.shape[0])
        acc2 = int(torch.sum(torch.argmax(logits2, dim=1) == targets2))/int(targets2.shape[0])        
        y_pred=(torch.argmax(logits1, dim=1).cpu().numpy()-1)*(-1)
        y_true=(targets1.cpu().numpy()-1)*(-1)
        f1=f1_score(y_true,y_pred)
        macro=f1_score(y_true, y_pred, average='macro')
        print('Test: acc | F1 | macro F1')
        print(acc1,f1,macro)   
        y_pred_mask=[y_pred[i] for i in mask]
        y_true_mask=[y_true[i] for i in mask]
        print('Test on Train mem: acc| F1| macro F1')
        print(accuracy_score(y_true_mask,y_pred_mask),f1_score(y_true_mask,y_pred_mask),f1_score(y_true_mask,y_pred_mask, average='macro'))
        
    return best_acc,best_model                 

if __name__ =='__main__':
    parser = argparse.ArgumentParser(
        description='HyperParameters for String Embedding')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--proximity', type=bool, default=False,
                        help='whether to use unsupervised proximity loss')
    parser.add_argument('--unsup_loss', type=str, default='normal',
                        help='kind of unsupervised loss')
    parser.add_argument('--cache_path',type=str, default='./walker_cache/',
                        help='cache path')    
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--cuda', type=int, default=3,
                    help='cuda device')
    parser.add_argument('--train_time_begin', type=int, default=2015,
                    help='time begin of the trainset')
    parser.add_argument('--train_time_end', type=int, default=2016,
                    help='time end of the trainset')
    parser.add_argument('--val_time_begin', type=int, default=2015,
                    help='time begin of the valset')
    parser.add_argument('--val_time_end', type=int, default=2016,
                    help='time end of the valset')    
    parser.add_argument('--test_time_begin', type=int, default=2017,
                    help='time begin of the testset')
    parser.add_argument('--test_time_end', type=int, default=2017,
                    help='time end of the testset')  
    parser.add_argument('--ratios', type=str, default='1 10 10',
                    help='ratios of the losses')         
    args = parser.parse_known_args()[0]
    cuda0 = torch.device('cuda:{}'.format(args.cuda))
    #load data
    train_time_begin, train_time_end = args.train_time_begin, args.train_time_end
    val_time_begin, val_time_end = args.val_time_begin, args.val_time_end
    test_time_begin, test_time_end = args.test_time_begin, args.test_time_end
    vote_dict = {'Yea': 0, 'Aye': 0, 'Nay': 1,
                 'No': 1}
  
    node_train=str(train_time_begin)+'_'+str(train_time_end)+'_node.txt'
    relation_train=str(train_time_begin)+'_'+str(train_time_end)+'_relation.txt'
    label_train=str(train_time_begin)+'_'+str(train_time_end)+'_label.txt'
    tag_label_train=str(train_time_begin)+'_'+str(train_time_end)+'_tag_label.txt'
    
    node_val=str(val_time_begin)+'_'+str(val_time_end)+'_node.txt'
    relation_val=str(val_time_begin)+'_'+str(val_time_end)+'_relation.txt'
    label_val=str(val_time_begin)+'_'+str(val_time_end)+'_label.txt'
    tag_label_val=str(val_time_begin)+'_'+str(val_time_end)+'_tag_label.txt'    

    node_test=str(test_time_begin)+'_'+str(test_time_end)+'_node.txt'
    relation_test=str(test_time_begin)+'_'+str(test_time_end)+'_relation.txt'
    label_test=str(test_time_begin)+'_'+str(test_time_end)+'_label.txt'
    tag_label_test=str(test_time_begin)+'_'+str(test_time_end)+'_tag_label.txt'
    best_acc,model=train_fn(args)





