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
Firstly, we process and number all nodes in the dataset. Then, for a given time period, run prepare.py to select corresponding nodes, compute relations and select labels for subsequent modelling. By running prepare.py, we can get txt files recording nodes, relations and labels of given period.  

#### Mode 2: Fine-tune the modules 

## Model




## Training
