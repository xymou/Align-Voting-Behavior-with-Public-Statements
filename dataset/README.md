# Twitter Dataset for Legislators

## Introduction
### Description
This dataset contains tweets posted by legislators. The overall crawling process is made up of 4 steps:
- Get personal pages of legislators from website of [US congress](http://www.congress.gov/)  
- Automatically crawl twitter account in their personal pages. (personal_web.py)  
- Manually search and identify legislators on Twitter, for those can not be acquired by step2
- Crawl tweets and following list using [twitterscraper](https://github.com/taspinar/twitterscraper)

Statistics of the dataset is shown as follows.  
![data](https://github.com/xymou/Align-Voting-Behavior-with-Public-Statements-for-Legislator-Representation-Learning/blob/main/dataset/data.png)

### Datafields
We got following information in our json files:
```
has_media: bool, does this tweet contains any media or not.  
hashtag: list, hashtags presented in this tweet.  
img_urls: list, urls of the images posted. 
is_replied: bool, is this tweet replied by others. 
is_reply_to: bool, is this tweet is a reply or not.  
likes: int, number of likes this tweet gets.  
links: list, urls contained in this tweet. 
parent_tweet_id: string, parent tweet id. 
replies: int, number of replies. 
reply_to_users: list, users replied by this tweet. 
retweets:int, number of retweets. 
screen_name: string, screen name of the author. 
text: string, text of the tweet.
text_html: string, text of the tweet in html style. 
timestamp: string, timestamp of the tweet, format: YYYY-MM-DDTHH:mm:ss. 
timestamp_epochs: int, timestamp in epochs.  
tweet_id: string, tweet_id. 
tweet_url: string, url of the tweet. 
user_id: string, user_id.
user_name: string, username.
vedio_url: string, url of the vedio.
```


## Downloads
1.Version_1  
It involves 735 legislators who vote between 2009 and 2018. All tweets and following lists of them before July 20th,2020 are crawled. This version of dataset is what we used in our paper. Each json file stores tweets of a legislator, and file 'objid2account.pkl' can help find the mapping relationship between the official id of legislators and their twitter account.

[download](https://www.dropbox.com/s/jq04755oj4akbea/Leg_Twitter_v1.zip?dl=0)  

2.Version_2  
It involves 843 legislators who vote between 2009 and 2020. All tweets of them before Dec 31st,2020 are crawled. Following list will be updated later.  
[download](https://www.dropbox.com/s/zfph6czv4og5ekk/Leg_Twitter_v2.zip?dl=0)

## Data Source and Tools
1.Twitter:https://twitter.com/  
3.twitterscraper: https://github.com/taspinar/twitterscraper  
4.US Congress: http://www.congress.gov/  
