# -*- coding: utf-8 -*-
"""
@author: xymou
"""
#preprocess the bill data and save pickle files


import pickle
import json
import pandas as pd
import os
import re

def save_obj(obj, name):
    with open('obj'+ name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

#%% get legislators involved from roll call data
filelist=['HS111_members.csv','HS112_members.csv','HS113_members.csv',
          'HS114_members.csv','HS115_members.csv']
members=[]
mem_lookup={}
ic2id={}

for file in filelist:
    df=pd.read_csv(file)
    mem=list(df['bioguide_id'].dropna().unique())
    members+=mem
    for i in range(len(df)):
        mem_lookup[df.loc[i]['bioguide_id']]=df.loc[i]['bioname']
        ic2id[df.loc[i]['icpsr']]=df.loc[i]['bioguide_id']
    
members=list(set(members))
party={}
state={}
state_list=list(df['state_abbrev'].dropna().unique())
state_dict={s:state_list.index(s) for s in state_list}
for file in filelist:
    df=pd.read_csv(file)
    for i in range(len(df)):
        if df.loc[i]['party_code']==100:
            party[df.loc[i]['bioguide_id']]=1
        elif df.loc[i]['party_code']==200:
            party[df.loc[i]['bioguide_id']]=2
        else:
            party[df.loc[i]['bioguide_id']]=0
        if df.loc[i]['state_abbrev'] in state_dict:
            state[df.loc[i]['bioguide_id']]=state_dict[df.loc[i]['state_abbrev']]

#%% prepare vote data
vote_dict={1:0,2:0,3:0,4:1,5:1,6:1}
def get_vote(congress='115',path1='HS115_rollcalls.csv',path2='HS115_votes.csv'):
    num2bill={}
    df=pd.read_csv(path1)
    for i in range(len(df)):
        b=df['bill_number'][i]
        if type(b)==str and len(re.findall('HRES|HR|HJRES|HCONRES|SRES|SJRES|SCONRES|S', b)):
            num2bill[df['chamber'][i]+str(df['rollnumber'][i])]=congress+b
    
    df=pd.read_csv(path2)
    vote={}
    for num in num2bill:
        num2=re.findall('[0-9]+',num)[0]
        num1=num[:-len(num2)]
        df2=df[(df['chamber']== num1) & (df['rollnumber']==int(num2))].reset_index(drop=True)
        tmp={}
        for i in range(len(df2)):
            if df2['cast_code'][i] in vote_dict:
                tmp[ic2id[df2['icpsr'][i]]]=vote_dict[df2['cast_code'][i]]
        vote[num2bill[num]]=tmp
    return vote

vote=get_vote('111',path1='HS111_rollcalls.csv',path2='HS111_votes.csv')
vote.update(get_vote('112',path1='HS112_rollcalls.csv',path2='HS112_votes.csv'))
vote.update(get_vote('113',path1='HS113_rollcalls.csv',path2='HS113_votes.csv'))  
vote.update(get_vote('114',path1='HS114_rollcalls.csv',path2='HS114_votes.csv')) 
vote.update(get_vote('115',path1='HS115_rollcalls.csv',path2='HS115_votes.csv')) 
save_obj(vote,'vote')


#%% save bill's year
filelist=['HS111_rollcalls.csv','HS112_rollcalls.csv','HS113_rollcalls.csv',
          'HS114_rollcalls.csv','HS115_rollcalls.csv']

bill2year={}
for file in filelist:
    c=file[2:5]
    df=pd.read_csv(file)
    for i in range(len(df)):
        b=df['bill_number'][i]
        if type(b)==str and len(re.findall('HRES|HR|HJRES|HCONRES|SRES|SJRES|SCONRES|S', b)):
            year=df['date'][i][:4]
            bill2year[c+b]=int(year)

save_obj(bill2year,'bill2year')

#%% save bill's title and description
filelist=['HS111_rollcalls.csv','HS112_rollcalls.csv','HS113_rollcalls.csv',
          'HS114_rollcalls.csv','HS115_rollcalls.csv']
billtext={}
for file in filelist:
    c=file[2:5]
    df=pd.read_csv(file)
    for i in range(len(df)):
        b=df['bill_number'][i]
        if type(b)==str and len(re.findall('HRES|HR|HJRES|HCONRES|SRES|SJRES|SCONRES|S', b)):
            billtext[c+b]=df['vote_desc'][i]

#the description is not detailed enough, so we use those in dataset of Yang et al.,
#use bill number to find corresponding information
billdesc=load_obj('billdesc')
bill2text={}
count=0
for key in billtext:
    if key in billdesc and len(billdesc[key])>1:
        count+=1
        bill2text[key]=billdesc[key]
    elif type(billtext[key])==str and len(billtext[key])>1:
        bill2text[key]=billtext[key]

save_obj(billtext,'bill2text')    