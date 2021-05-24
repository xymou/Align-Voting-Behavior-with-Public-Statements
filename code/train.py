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
from model import *
from walker import *

def train_fn(args):
    cs_loss=nn.CrossEntropyLoss()
    random_shuffle(node_train, relation_train, label_train, tag_label_train)
    random_shuffle(node_test, relation_test, label_test, tag_label_test)  
    
    #prepare data
    label_train1,label_val=prepare(label_train)
    train=mydata(6166,768)
    X, Y, Z, mem2mem_matrix, mem2leg_matrix, mem2tag_matrix, leg2leg_matrix,\
                leg2tag_matrix, tag2tag_matrix,leg2mem_matrix, tag2mem_matrix, tag2leg_matrix=train.prepare(node_train, relation_train)
    train.batcher(128,label_train1,tag_label_train)
        
    with open(label_val) as fp:
        df=fp.readlines()
    fp.close()
    val=mydata(6166,768)
    X1, Y1, Z1, mem2mem_matrix1, mem2leg_matrix1, mem2tag_matrix1, leg2leg_matrix1,\
                leg2tag_matrix1, tag2tag_matrix1, leg2mem_matrix1, tag2mem_matrix1, tag2leg_matrix1= val.prepare(node_val, relation_val)
    val.batcher(len(df),label_val,tag_label_val)     
    
    with open(label_test) as fp:
        df=fp.readlines()
    fp.close()
    test=mydata(6166,768)
    X2, Y2, Z2, mem2mem_matrix2, mem2leg_matrix2, mem2tag_matrix2, leg2leg_matrix2,\
                leg2tag_matrix2, tag2tag_matrix2, leg2mem_matrix2, tag2mem_matrix2, tag2leg_matrix2= test.prepare(node_test, relation_test)
    test.batcher(len(df),label_test,tag_label_test) 
    
    
    model=HGCN(64,64,64,2,2).to(cuda0)
    
    walker=Walker(train.adj_lists,train.weight_lists,train.mem_nodes,train.mems, args.cache_path)
    #optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)     
    #Use different learning rates for different layers
    params = [
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
            logits1,logits2=model(X, Y, Z, mem2mem_matrix, mem2leg_matrix, mem2tag_matrix, leg2leg_matrix,\
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
                    logits1,logits2=model(X1, Y1, Z1, mem2mem_matrix1, mem2leg_matrix1, mem2tag_matrix1, leg2leg_matrix1,\
                            leg2tag_matrix1, tag2tag_matrix1, leg2mem_matrix1, tag2mem_matrix1, tag2leg_matrix1, mem, leg, mem2, tag)
                    acc1 = int(torch.sum(torch.argmax(logits1, dim=1) == targets1))/int(targets1.shape[0])
                    acc2 = int(torch.sum(torch.argmax(logits2, dim=1) == targets2))/int(targets2.shape[0])
                    y_pred=(torch.argmax(logits1, dim=1).cpu().numpy()-1)*(-1)
                    y_true=(targets1.cpu().numpy()-1)*(-1)
                    f1=f1_score(y_true,y_pred)
                    macro=f1_score(y_true, y_pred, average='macro')
                    print('Val: acc | F1 | macro F1')    
                    print(acc1,f1,macro)
                    #print(torch.argmax(logits, dim=1))
                    if best_acc<acc1:
                        best_acc=acc1
                        best_model=HGCN(64,64,64,2,2).to(cuda0)
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
        logits1,logits2=best_model(X1, Y1, Z1, mem2mem_matrix1, mem2leg_matrix1, mem2tag_matrix1, leg2leg_matrix1,\
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
        logits1,logits2=best_model(X2, Y2, Z2, mem2mem_matrix2, mem2leg_matrix2, mem2tag_matrix2, leg2leg_matrix2,\
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
    parser.add_argument('--cuda', type=int, default=0,
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

    