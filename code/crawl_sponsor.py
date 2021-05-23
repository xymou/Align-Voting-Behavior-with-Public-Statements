# -*- coding: utf-8 -*-
"""
@author: xymou
"""

#crawl sponsor list of given bill
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import json
import time
import json
import re
import pandas as pd
from tqdm import tqdm
import pickle
#%%
def save_obj(obj, name):
    with open('obj'+ name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

#%%
#get url of sponsor page of given bill
def bill2url(congress,bill):
    num=re.findall('[0-9]+', bill)[0]
    if bill.startswith('HRES'):
        domain='house-resolution'
    elif bill.startswith('HR'):
        domain='house-bill'
    elif bill.startswith('HJRES'):
        domain='house-joint-resolution'
    elif bill.startswith('HCONRES'):
        domain='house-concurrent-resolution'
    elif bill.startswith('SJRES'):
        domain='senate-joint-resolution'
    elif bill.startswith('SRES'):
        domain='senate-resolution'
    elif bill.startswith('SCONRES'):
        domain='senate-concurrent-resolution'
    elif bill.startswith('S'):
        domain='senate-bill'
    url='https://www.congress.gov/bill/'+str(congress)+'th-congress/'+domain+'/'+str(num)+'/cosponsors?searchResultViewType=expanded'
    return url

#get sponsor list    
def get_spon(url):
    res=requests.get(url)
    res=BeautifulSoup(res.text,features="html.parser")
    try:
        div=res.find_all("div",attrs={'class':"search-row inner-bill-detail"})[0]
        tb=div.find_all('td',attrs={'class':"actions"})
        data=[]
        for t in tb:
            tmp={}
            tmp['name']=t.text[1:-1]
            tmp['page']=re.findall('a href=".*?"',str(t))[0][8:-1]
            data.append(tmp)
    except:
        data=[]
    return data

#%% save these data as a {bill_num:data} dict
def crawl(congress,bill):
    data={}
    count=0
    for i in tqdm(range(len(bill))):
        time.sleep(1)
        if count % 10==1:
            time.sleep(5)
        if count % 100==1:
            time.sleep(30)
        c,b=congress[i],bill[i]
        url=bill2url(c,b)
        tmp=get_spon(url)
        data[c+b]=tmp
        count+=1
    save_obj(data,'sponsor116')
    return data

#%% get bill number from the csv files of voteview
filelist=['HS111_rollcalls.csv','HS112_rollcalls.csv','HS113_rollcalls.csv',
          'HS114_rollcalls.csv','HS115_rollcalls.csv',]
bill=[]
congress=[]

for file in filelist:
    df=pd.read_csv(file)
    b=list(df['bill_number'].dropna().unique())
    b=[t for t in b if len(re.findall('HRES|HR|HJRES|HCONRES|SRES|SJRES|SCONRES|S', t))]
    bill+=b
    congress+=[file[2:5]]*len(b)

#%% run
data=crawl(congress,bill)
data={}
for key in sponsor:
    if len(sponsor[key])==0:
        data[key]=[]
    else:
        tmp=[]
        for item in sponsor[key]:
            tmp.append(item['page'].split('/')[-1])
        data[key]=tmp
save_obj(data,'sponsor')
