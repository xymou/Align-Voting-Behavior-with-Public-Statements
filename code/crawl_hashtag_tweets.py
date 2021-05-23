# -*- coding: utf-8 -*-
"""
@author: xymou
"""

import json
from twitterscraper import query_tweets
import os
from tqdm import tqdm
import threading
import traceback
import time

class myThread (threading.Thread):
    def __init__(self, hashtag, id_):
        threading.Thread.__init__(self)
        self.hashtag = hashtag
        self.id = id_
    def run(self):
        os.system('twitterscraper ' + '#' + self.hashtag + ' -l 200 -p 1 --lang en -ow -o '+str(self.id)+'.json')

if __name__ == '__main__':
    unique_hashtag = ['obamacare','dreamer']
    hashtag_data = {}
    if os.path.exists('.\hashtag_data.json'):
        with open('hashtag_data.json') as f:
            hashtag_data = json.load(f)
    num_thread = 10
    for i in tqdm(range(0,len(unique_hashtag),num_thread)):
        if len(unique_hashtag) - i < num_thread:
            num_thread_now = len(unique_hashtag) - i
        else:
            num_thread_now = num_thread
        is_continue = True
        for j in range(num_thread_now):
            if unique_hashtag[i+j] not in hashtag_data:
                is_continue = False
        if is_continue:
            continue
        threads = [myThread(unique_hashtag[i+j],j) for j in range(num_thread_now)]
        for t in threads:
            t.start()
            time.sleep(0.15)
        for t in threads:
            t.join()
        for j in range(num_thread_now):
            try:
                with open(str(j)+'.json') as f:
                    tweets = json.load(f)
                hashtag_data[unique_hashtag[i+j]] = tweets
            except:
                traceback.print_exc()
        with open('hashtag_data.json',"w") as f:
            json.dump(hashtag_data,f)

    
            
        