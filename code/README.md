# Code 

## Preparation
To avoid constructing graphs repeatedly and wasting time, we store information of nodes and relations for each time period in advance. 
### Data Preprocessing
**1. Prepare bill data**  
Our bill data comes from [Voteview](https://voteview.com/data) and [Yang et al.,](http://www.sdspeople.fudan.edu.cn/zywei/data/fudan-USRollCall.zip). To reserve bills and resolutions and convert votes into unified labels, we need to process the original csv files. You can find this in preprocess.py.  
After preprocessing, bill-related data and basic information of legislators is stored in pickle files named vote, bill2text, bill2year, mem2state, mem2party.

**2. Prepare sponsor data**  
We crawl sponsor list of each legislation in congress.gov. Code can be found in crawl_sponsor.py.  

**3. Prepare Twitter data**  
We have provided our Twitter data link in the dataset folder. Just download it and place the json files and pickle file id2account in the /data/all_tweets/ folder under this path.  
Besides, for each hashtag, we need to get tweets with the tag on Twitter platform, to get to know the content the tag expresses. crawl_hashtag_tweets.py is an example to achieve this goal.

### Graph Information Preparation  
#### Mode 1: Use pretrained embeddings and save
Firstly, we process and number all nodes in the dataset. Then, for a given time period, run prepare.py to select corresponding nodes, compute relations and select labels for subsequent modelling. By running prepare.py, we can get txt files recording nodes, relations and labels of given period. Since the size of the txt files exceeds limit of github, we don't provide them here.  
This mode is faster than mode 2, since we do not fine-tune the encoders.  
The training file is train.py below.

#### Mode 2: Fine-tune the modules 
Fine tune Bert and member encoders.  
The training file is fine_tune.py below. 

## Model
model.py involves some tool functions and RGCN model;   
walker.py is used for sampling positive and negative neighbors for proximity loss.


## Training
#### Mode 1
train.py includes process of data loading, model construction and training. The parameters include:  
```
--epochs: number of epochs to train (default: 100)  
--proximity: whether to use proximity loss (default: False)
--unsup_loss: kind of unsupervised loss (default: normal)  
--cache_path: cache path  
--lr: learning rate (default: 0.0001)
--cuda: cuda service (default: 0)
--train_time_begin: time begin of the trainset (default: 2015)
--train_time_end: time end of the trainset (default: 2016)
--val_time_begin: time begin of the valset (default: 2015)
--val_time_end: time end of the valset (default: 2016)
--test_time_begin: time begin of the testset (default: 2017)
--test_time_end: time end of the testset (default: 2018)
--ratios: ratios of the losses (default: '1 10 10')
```
#### Mode 2
fine_tune.py combines something of the prepare.py and train.py. The parameters are the same as train.py.
