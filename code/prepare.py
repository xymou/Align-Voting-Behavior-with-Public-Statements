# -*- coding: utf-8 -*-
"""
@author: xymou
"""

import json
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import pickle
from tqdm import tqdm
from scipy.spatial.distance import pdist
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import random
import argparse
import math
import copy
stws = stopwords.words('english')
stws+=['[URL]','[MENTION]','[PIC]','[url]','[mention]','[pic]','w/','$','rt','via','--','&','[url]…','-']
stws+=[str(i) for i in list(range(10))]
stws+= [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '/', '-',
                        '<', '>','’']

def filename(dir):
    namelist=[]
    for root,dirs,files in os.walk(dir):
        namelist.append(files)
    return namelist[0]

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


###### overall preparation
#prepare BERT client
#starting your bert service at the terminal first
#e.g. by running command: bert-serving-start -model_dir /remote-home/my/uncased_L-12_H-768_A-12 -max_seq_len 512 -num_worker 4
from bert_serving.client import BertClient
bert=BertClient()

#initialize legislation embedding
def leg_encode(time_begin,time_end):
    #find the legislations from 2009 to 2018
    vote=load_obj('vote','./data/')
    bill2text=load_obj('bill2text','./data/')
    lookup=load_obj('bill2year','./data/')
    emb=[]
    #encode the text
    legislation_list=list(bill2text.keys())
    for bill in tqdm(legislation_list):  
        text =bill2text[bill]
        rep=bert.encode([text])[0].reshape(1,-1)
        emb.append(rep)
    emb=np.concatenate(emb,axis=0)
    return legislation_list,emb,lookup


def clean(text): #clean tweets
    pattern1='(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
    pattern2='@([^@ ]*)'
    pattern3='pic.twitter.com/.*'
    text=re.sub(pattern1,' [URL]',text)
    text=re.sub(pattern2,' [MENTION]',text)
    text=re.sub(pattern3,' [PIC]',text)
    text=re.sub('\xa0','',text)
    return text   

#initialize hashtag embedding
def tag_twi(tag,k=50,time_end='2019-01-01'): #randomly choose K tweets with the given hashtag
    res=[]
    with open('./data/tags/'+tag[1:-1]+'.json') as f: #json file saving tweets on Twitter platform with specific hashtag
        data=json.load(f)
    f.close()
    for d in data:
        if d['timestamp'][:10] <time_end:
            text=d['text'].lower()
            res.append(text)
    if len(res)>k:
        res=random.sample(res,k)
    res=[clean(r) for r in res]
    return res      

def tag_encode(tag_list,tag2year,time_begin, time_end):
    emb=[]
    for tag in tqdm(tag_list):
        if tag2year[tag]>=time_begin and tag2year[tag]<=time_end:
            #randomly select tweets under this hashtag
            texts=tag_twi(tag,50)
            #encode and average
            texts=np.mean(bert.encode(texts),axis=0).reshape(1,-1)
            rep=texts
            emb.append(rep)
    emb=np.concatenate(emb,axis=0)
    return emb

#initialize legislator embedding
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

def mem_encode(legislation_list):
    vote=load_obj('vote','./data/')
    bill2year=load_obj('bill2year','./data/')
    mem2state=load_obj('mem2state','./data/')
    mem2party=load_obj('mem2party','./data/')
    member_list=[m for m in list(mem2party.keys()) if type(m)==str]
    state_size=len(list(set(mem2state.values())))
    party_size=len(list(set(mem2party.values())))
    mem_lookup={} #the first time a legislator appear in roll call votes
    for leg_name in tqdm(legislation_list):
        year = bill2year[leg_name]
        for member in vote[leg_name]:
            if member not in mem_lookup or year<mem_lookup[member]:
                mem_lookup[member]=year
    member_all=np.array(range(len(member_list)))
    state_all=np.array([mem2state[m] for m in member_list])
    party_all=np.array([mem2party[m] for m in member_list])
    member_all=torch.tensor(member_all, dtype=torch.long)
    state_all=torch.tensor(state_all, dtype=torch.long)
    party_all=torch.tensor(party_all, dtype=torch.long)
    encoder=mem_encoder(len(member_all),state_size,party_size)
    X=encoder.encode(member_all,state_all,party_all)
    return member_list,X.detach().numpy(), mem_lookup

#save all the nodes in advance, including their initial embedding, their type(mem/leg/tag), their info(bill number for leg, member ID for mem and tag) and their time
def prepare_nodes(time_begin=2009, time_end=2018, load=True):
    if load:
        node2emb=load_obj('node2emb',None)
        node2type=load_obj('node2type',None)
        node2info=load_obj('node2info',None)
        node2year=load_obj('node2year',None)
        return node2emb, node2type, node2info, node2year
    legislation_list,leg_emb,leg_lookup=leg_encode(time_begin, time_end)
    mem_list,mem_emb, mem_lookup=mem_encode(legislation_list)
    tag_list=load_obj('taglist')
    tag_list=list(tag_list.keys())
    tag2year=load_obj('tag2year')
    tag_emb=tag_encode(tag_list,tag2year, time_begin, time_end)
    tag_lookup=tag2year
    node2type={}
    node2info={}
    node2year={} #the first time when a node appear in context
    node2emb={}
    for i in range(len(mem_list)):
        node2emb[i]=mem_emb[i]
        node2type[i]='mem'
        node2info[i]=mem_list[i]
        node2year[i]=mem_lookup[mem_list[i]]
    k=i+1
    for i in range(len(legislation_list)):
        node2emb[k+i]=leg_emb[i]
        node2type[k+i]='leg'
        node2info[k+i]=legislation_list[i]
        node2year[k+i]=leg_lookup[legislation_list[i]]
    k=k+i+1
    for i in range(len(tag_list)):
        node2emb[k+i]=tag_emb[i]
        node2type[k+i]='tag'
        node2info[k+i]=tag_list[i]
        node2year[k+i]=tag2year[tag_list[i]]
    with open('node.txt','w') as fp:
        for node in node2emb:
            line=str(node)+' '+" ".join(" ".join(map(str,node2emb[node].tolist()))[1:-1].split(','))
            fp.write(line+'\n')
    fp.close()
    save_obj(node2emb,'node2emb')
    save_obj(node2type,'node2type')
    save_obj(node2info,'node2info')
    save_obj(node2year,'node2year')
    return node2emb, node2type, node2info, node2year

#this process [prepare_nodes(time_begin=2009, time_end=2018, load=False)] needs to be run only once; 
#after running once, you can load the data directly and run prepare_nodes(time_begin=2009, time_end=2018, load=True)

###### creating graph for a given time period: select nodes, compute relations, save labels...

#select nodes
#example: 
#select nodes of 112th congress for training: select_node(2011,2012,...)
#select nodes of 113th congress for testing: select(2011,2014,...) that's to say, you need to include some historical information
#select nodes of 2013 for testing: select(2011,2013,...)
def select_node(time_begin,time_end,node2emb,node2type, node2info, node2year):
    nodes=[]
    with open(str(time_begin)+'_'+str(time_end)+'_node.txt','w') as fp:
        for node in node2emb:
            if (node2year[node]<=time_end and node2type[node]=='mem') or (node2year[node]<=time_end and node2year[node]>=time_begin and node2type!='mem'):
                line=str(node)+' '+" ".join(" ".join(map(str,node2emb[node].tolist()))[1:-1].split(','))
                fp.write(line+'\n')
                nodes.append(node)             
    fp.close()
    return nodes

def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

def refer_num(member,tag,time_end):
    #time a legislator mentioned a hashtag; need to consider those without available accounts
    member2account=load_obj('id2account','./data/')
    if member in member2account:
        member=member2account[member]    
    else:
        return 0
    path='./data/all_tweets/'
    try:
        with open(path+member+'.json') as f:
            data=json.load(f)
        f.close()
    except:
        return 0
    count=0
    for d in data:
        if d['timestamp'][:10] < str(time_end)+'-01-01':
            if tag in d['text'].lower():
                count+=1
    return count

def relevance(leg,tag):
    #compute semantic similarity of legislation and hashtag
    bill2text=load_obj('bill2text','./data/')
    text=bill2text[leg]
    legtext=set([x for x in nltk.word_tokenize(text.lower()) if x not in stws])
    res=tag_twi(tag)
    tagtext=[]
    for r in res:
        tagtext+=[x for x in nltk.word_tokenize(r.lower()) if x not in stws]
    tagtext=set(tagtext)
    return len(tagtext & legtext)

def leg_relevance(leg1,leg2):
    #compute semantic similarity of two legislation
    bill2text=load_obj('bill2text','./data/')
    text1=bill2text[leg1]
    legtext1=set([x for x in nltk.word_tokenize(text1.lower()) if x not in stws])
    text2=bill2text[leg2]
    legtext2=set([x for x in nltk.word_tokenize(text2.lower()) if x not in stws])
    return len(legtext1 & legtext2)

#select relations
#to avoid information leakage, when we predict, we load all involved nodes, but only use historical information to get relations
#example
#select relations of 112th congress for training: select(2011,2012,2011,...)
#select relations of 113th congress for testing: select(2011,2014,2013,...)
#select relations of 2013 for testing: select(2011,2013,2013,...)
def select_relation(time_begin, time_end, relation_end, node2emb, node2type, node2info, node2year, write=False, test=False, eliminate=True):
    #select nodes first
    nodes=select_node(time_begin, time_end, node2emb, node2type, node2info, node2year)
    vote=load_obj('vote','./data/')
    bill2year=load_obj('bill2year','./data/')
    sponsor=load_obj('sponsorall','./data/')
    source=[]
    end=[]
    weights=[]
    rtypes=[]
    mem=[node for node in nodes if node2type[node]=='mem']
    mem_dict={m:mem.index(m) for m in mem}
    info2node={v:k for k,v in node2info.items()}
    adj_matrix=np.zeros((len(mem),len(mem)))
    leg_nodes=[node for node in nodes if node2type[node]=='leg']
    tag_nodes=[node for node in nodes if node2type[node]=='tag']
    leg_dict={l:leg_nodes.index(l) for l in leg_nodes}
    tag_dict={t:tag_nodes.index(t) for t in tag_nodes}
    mem2leg_adj=np.zeros((len(mem),len(leg_nodes)))
    leg_adj=np.zeros((len(leg_nodes),len(leg_nodes)))
    tag_adj=np.zeros((len(tag_nodes),len(tag_nodes)))
    leg2tag_adj=np.zeros((len(leg_nodes),len(tag_nodes)))
    mem2tag_adj=np.zeros((len(mem),len(tag_nodes)))
    
    #mem2mem relation: co-sponsor
    for bill in sponsor:
        if int(bill2year[bill])<=relation_end:
            for i in range(len(sponsor[bill])):
                spon=sponsor[bill][i]
                if spon in info2node and bill in info2node and info2node[spon] in mem_dict and info2node[bill] in leg_dict:
                    mem2leg_adj[mem_dict[info2node[spon]]][leg_dict[info2node[bill]]]=1
                for j in range(len(sponsor[bill])):
                    temp1=sponsor[bill][i]
                    temp2=sponsor[bill][j]
                    if temp1 in info2node and temp2 in info2node and info2node[temp1] in mem_dict and info2node[temp2] in mem_dict:
                        adj_matrix[mem_dict[info2node[temp1]]][mem_dict[info2node[temp2]]]+=1                   
                          
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[i])):
            if adj_matrix[i][j]!=0:
                source.append(mem[i])
                end.append(mem[j])
                weights.append(adj_matrix[i][j])
                rtypes.append('mem2mem')   
    print('mem2mem finished!')
    for i in range(len(mem2leg_adj)):
        for j in range(len(mem2leg_adj[i])):
            if mem2leg_adj[i][j]!=0:
                source.append(mem[i])
                end.append(leg_nodes[j])
                weights.append(mem2leg_adj[i][j])
                rtypes.append('mem2leg')                              
    print('mem2leg finished!')       
     
    
    #leg2leg relation: semantic similarity
    for node1 in leg_nodes:
        for node2 in leg_nodes:
            if node1!=node2:
                leg_adj[leg_dict[node1]][leg_dict[node2]]=leg_relevance(node2info[node1],node2info[node2])  
    for i in range(len(leg_adj)):
        for j in range(len(leg_adj[i])):
            if leg_adj[i][j]!=0:
                source.append(leg_nodes[i])
                end.append(leg_nodes[j])
                weights.append(leg_adj[i][j])
                rtypes.append('leg2leg')
    print('leg2leg finished!')      
    
    #tag2tag relation: Co-occurrence 
    path='./data/all_tweets/'
    filelist=filename(path)
    for file in tqdm(filelist):
        with open(path+file) as f:
            data=json.load(f)
        f.close()
        for d in data:
            if d['timestamp'][:10] < str(relation_end)+'-01-01':
                ts=re.findall('\#.+? ',d['text'].lower())
                for m in ts:
                    for n in ts:
                        m=m.strip()+' '
                        n=n.strip()+' '
                        if m!=n and m in info2node and n in info2node and info2node[m] in tag_dict and info2node[n] in tag_dict:
                            tag_adj[tag_dict[info2node[m]]][tag_dict[info2node[n]]]+=1
    
    for i in range(len(tag_adj)):
        for j in range(len(tag_adj[i])):
            if tag_adj[i][j]!=0:
                source.append(tag_nodes[i])
                end.append(tag_nodes[j])
                weights.append(tag_adj[i][j])
                rtypes.append('tag2tag')
    print('tag2tag finshed!')
    
    #leg2tag relation: semantic similarity
    for node1 in tqdm(leg_nodes):
        for node2 in tag_nodes:
            leg2tag_adj[leg_dict[node1]][tag_dict[node2]]=relevance(node2info[node1],node2info[node2])              
    for i in range(len(leg2tag_adj)):
        for j in range(len(leg2tag_adj[i])):
            if leg2tag_adj[i][j]!=0:
                source.append(leg_nodes[i])
                end.append(tag_nodes[j])
                weights.append(leg2tag_adj[i][j])
                rtypes.append('leg2tag')  
    print('leg2tag finished!')      
    
    #mem2tag relation: # of reference
    for node1 in tqdm(tag_nodes):
        for node2 in mem:
            count=refer_num(node2info[node2],node2info[node1],relation_end)
            mem2tag_adj[mem_dict[node2]][tag_dict[node1]]=count  
    for i in range(len(mem2tag_adj)):
        for j in range(len(mem2tag_adj[i])):
            if mem2tag_adj[i][j]!=0:
                source.append(mem[i])
                end.append(tag_nodes[j])
                weights.append(mem2tag_adj[i][j])
                rtypes.append('mem2tag')
    print('mem2tag finished!')     
    
    if write:
        path=str(time_begin)+'_'+str(time_end)+'_relation.txt'
        with open(path,'w') as fp:
            for i in range(len(source)):
                line=str(source[i])+' '+str(end[i])+' '+str(weights[i])+' '+rtypes[i]
                fp.write(line+'\n')
        fp.close()
    return source, end, weights, rtypes

#prepare labels
#example
#select labels of 112th congress for training: select(2011,2012,2011,2012,...)
#select labels of 113th congress for testing: select(2011,2014,2013, 2014,...)
#select labels of 2013 for testingL select(2011,2013,2013,2013,...)
def select_labels(time_begin, time_end, label_begin, label_end, node2emb, node2type, node2info, node2year, write=True, test=False):
    #select nodes first
    nodes=select_node(time_begin, time_end, node2emb, node2type, node2info, node2year)
    vote=load_obj('vote','./data/')
    info2node={v:k for k,v in node2info.items()}
    source=[]
    end=[]
    labels=[]
    #when testing, nodes are all loaded, but only labels those need to be predicted will be selected(labels of training set will not be loaded)
    leg_nodes=[node for node in nodes if node2type[node]=='leg' and node2year[node]>=label_begin and node2year[node]<=label_end]
    for node in leg_nodes:
        bill= node2info[node]
        for member in vote[bill]:
            if member not in info2node:
                continue
            source.append(info2node[member])
            end.append(node)
            labels.append(vote[bill][member])
    if write:
        with open(str(time_begin)+'_'+str(time_end)+'_label.txt','w') as fp:
            for i in range(len(source)):
                line=str(source[i])+' '+str(end[i])+' '+str(labels[i])
                fp.write(line+'\n')
        fp.close()
        print('-----label finished-----')
    return source, end, labels

def select_tag_labels(time_begin, time_end, label_begin, label_end, node2emb, node2type, node2info, node2year, write=True, test=False):
    #select nodes first
    nodes=select_node(time_begin, time_end, node2emb, node2type, node2info, node2year)
    tag_dataset=load_obj('tag_dataset','./data/')
    info2node={v:k for k,v in node2info.items()}
    source=[]
    end=[]
    labels=[]
    
    tag_nodes=[node for node in nodes if node2type[node]=='tag' and node2year[node]>=label_begin and node2year[node]<=label_end]
    for node in tag_nodes:
        tag= node2info[node]
        if tag not in tag_dataset:
            continue
        data=tag_dataset[tag]
        for key in data:
            if key in info2node and info2node[key] in nodes:
                source.append(info2node[key])
                end.append(node)
                labels.append(tag_dataset[tag][key])
    if write:
        with open(str(time_begin)+'_'+str(time_end)+'_tag_label.txt','w') as fp:
            for i in range(len(source)):
                line=str(source[i])+' '+str(end[i])+' '+str(labels[i])
                fp.write(line+'\n')
        fp.close()
        print('-----tag label finished-----')
    return source, end, labels

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='HyperParameters for String Embedding')
    parser.add_argument('--time_begin', type=int, default=2011,
                        help='time begin')
    parser.add_argument('--time_end', type=int, default=2012,
                        help='time end')     
    parser.add_argument('--relation_time_end', type=int, default=2011,
                        help='relation time end')    
    parser.add_argument('--label_time_begin', type=int, default=2011,
                        help='label time begin')                                               
    parser.add_argument('--label_time_end', type=int, default=2012,
                        help='label time end')                                                 
    parser.add_argument('--write', type=bool, default=True,
                        help='write the file')       
    parser.add_argument('--test', type=bool, default=False,
                        help='whether to prepare a test data')      
    args = parser.parse_known_args()[0]
    
    node2emb, node2type, node2info, node2year=prepare_nodes( time_begin=args.time_begin, time_end=args.time_end, load=True)
    nodes=select_node(args.time_begin,args.time_end,node2emb,node2type, node2info, node2year)
    source, end, weights, rtypes=select_relation(args.time_begin, args.time_end, args.relation_time_end, node2emb, node2type, node2info, node2year, write=args.write, test=args.test)
    source, end, labels=select_labels(args.time_begin, args.time_end, args.label_time_begin, args.label_time_end, node2emb, node2type, node2info, node2year, write=args.write, test=args.test)
    source, end, labels=select_tag_labels(args.time_begin, args.time_end, args.label_time_begin, args.label_time_end, node2emb, node2type, node2info, node2year, write=args.write,test=args.test)