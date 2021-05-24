# -*- coding: utf-8 -*-
"""
@author: xymou
This code is partly learned from FANG: Leveraging Social Context for Fake News Detection Using Graph Representation
"""
import numpy as np
from scipy import special
import random
import torch
import torch.nn.functional as F
from torch import nn
import time
import pickle
from collections import OrderedDict
import os

LOGGING, LOG_PROB = False, 1
CACHING = True


def load_from_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_to_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

class LRUCache:

    # initialising capacity
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: int, value: set()) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

class Walker(object):
    def __init__(self, adj_lists, weight_lists,mems, mems_train, cache_path, max_engages=100):
        super(Walker,self).__init__()
        self.Q=10
        self.MEM_WALK_LEN=1 
        self.MEM_NUM_POS=5
        self.MEM_NUM_NEG=5      
        self.MARGIN=3
        self.max_engages = max_engages
        self.adj_lists, self.weight_lists=adj_lists, weight_lists
        self.mems=mems
        self.mems_train=mems_train
        self.sampled_nodes = list(set(self.mems))
    
        self.positive_pairs = []
        self.negative_pairs = []
        self.node_positive_pairs = {}
        self.node_negative_pairs = {}
        self.unique_nodes_batch = []
        
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)  
        
        self.mem_cache_path = cache_path+"mem_" + str(self.MEM_WALK_LEN)
        
        self.mem_neighbor_cache, self.write_cache = self.load_cache_far_nodes()        
        
        
    def save(self):
        if self.write_cache and CACHING:
            print("Saving cache")       
            save_to_pickle((self.mem_neighbor_cache.cache,self.mem_neighbor_cache.capacity),self.mem_cache_path)          
        
    def load_cache_far_nodes(self):
        if CACHING and os.path.exists(self.mem_cache_path):
            print("loading cache")
            mem_neighbor_cache_dict, mem_cache_capacity = load_from_pickle(self.mem_cache_path)
                 
            mem_neighbor_cache = LRUCache(mem_cache_capacity)
            mem_neighbor_cache.cache = mem_neighbor_cache_dict 
            write_cache = False
        else:
            capacity = 20000
            print("Creating new far node cache with capacity {}".format(capacity))
            mem_neighbor_cache= LRUCache(capacity)
            write_cache = True            
        return mem_neighbor_cache, write_cache           
    
    
    def compute_cs_loss(self,logits,label):
        cs_loss = self.cross_entropy(logits, label)
        return cs_loss
    
    def compute_unsup_loss_normal(self, embeddings, nodes):
        node2index = {n: i for i, n in enumerate(nodes)}
        node_scores = []
        
        for node in self.node_positive_pairs:
            pps=self.node_positive_pairs[node]  
            nps=self.node_negative_pairs[node]
            if len(pps) == 0 or len(nps) == 0:
                continue
            
            # Q * Exception(negative score)
            neg_indexs = [list(x) for x in zip(*nps)]
            neg_node_indexs = [node2index[x] for x in neg_indexs[0]]   
            neg_neighb_indexs = [node2index[x] for x in neg_indexs[1]]
            neg_node_embeddings, neg_neighb_embeddings = embeddings[neg_node_indexs], embeddings[neg_neighb_indexs]
            neg_sim_score = F.cosine_similarity(neg_node_embeddings, neg_neighb_embeddings)
            neg_score = self.Q * torch.mean(torch.log(torch.sigmoid(-neg_sim_score)+1e-10), 0)
            
            # multiple positive score
            pos_indexs = [list(x) for x in zip(*pps)]
            pos_node_indexs=[node2index[x] for x in pos_indexs[0]]
            pos_neighb_indexs=[node2index[x] for x in pos_indexs[1]]
            pos_node_embeddings, pos_neighb_embeddings = embeddings[pos_node_indexs], embeddings[pos_neighb_indexs]
            pos_sim_score = F.cosine_similarity(pos_node_embeddings, pos_neighb_embeddings)
            pos_score = torch.log(torch.sigmoid(pos_sim_score)+1e-10)
            
            # proximity loss
            node_score = torch.mean(- pos_score - neg_score).view(1, -1)
            node_scores.append(node_score)
            
        if len(node_scores) > 0:
            loss = torch.mean(torch.cat(node_scores, 0))
        else:
            loss = None
        return loss            
    
    def compute_unsup_loss_margin(self,embeddings, nodes):
        node2index = {n: i for i, n in enumerate(nodes)}
        nodes_score = []
        assert len(self.node_positive_pairs) == len(self.node_negative_pairs)
        
        for node in nodes:
            pps = self.node_positive_pairs[node]
            nps = self.node_negative_pairs[node]
            if len(pps) == 0 or len(nps) == 0:
                continue
            
            indexs = [list(x) for x in zip(*pps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            pos_score, _ = torch.min(torch.log(torch.sigmoid(pos_score)+1e-10), 0)
            
            indexs = [list(x) for x in zip(*nps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            neg_score, _ = torch.max(torch.log(torch.sigmoid(neg_score)+1e-10), 0)
            nodes_score.append(
                torch.max(torch.tensor(0.0).to(self.device), neg_score - pos_score + self.MARGIN).view(1, -1))            
        if len(nodes_score) > 0:
            loss = torch.mean(torch.cat(nodes_score, 0), 0)
        else:
            loss = None

        return loss
    
    def extend_nodes(self, nodes):
        self.positive_pairs = []
        self.node_positive_pairs = {}
        self.negative_pairs = []
        self.node_negative_pairs = {}
        
        mem_nodes = [node for node in nodes if node in self.mems]
        self.get_proximity_samples(mem_nodes, self.MEM_NUM_POS, self.MEM_NUM_NEG,
                        self.MEM_WALK_LEN, self.mem_neighbor_cache)         
    
        self.unique_nodes_batch = list(
            set([i for x in self.positive_pairs for i in x]) | set([i for x in self.negative_pairs for i in x]))
        return self.unique_nodes_batch          
    
    
    def reset(self):
        self.positive_pairs = []
        self.node_positive_pairs = {}
        self.negative_pairs = []
        self.node_negative_pairs = {}    
    
    def get_proximity_samples(self, nodes, num_pos, num_neg, neg_walk_len, neighbor_cache):
        # for optimization,  perform both positive and negative sampling in a single walk
        mem=set(self.mems)
        for node in nodes:
            homo_nodes = mem
            neighbors = neighbor_cache.get(node)
            if neighbors==-1:
                neighbors, frontier = {node}, {node}
                for _ in range(neg_walk_len):
                    current = set()
                    for outer in frontier:
                        current |= set(self.get_neigh_weights(outer, node_only=True))
                    frontier = current - neighbors
                    neighbors |= current
                if CACHING:
                    neighbor_cache.put(node, neighbors)
            far_nodes = homo_nodes - neighbors
            neighbors -= {node}
            
            # Update positive samples
            pos_samples = random.sample(neighbors, num_pos) if num_pos < len(neighbors) else neighbors
            pos_pairs = [(node, pos_node) for pos_node in pos_samples]
            self.positive_pairs.extend(pos_pairs)
            self.node_positive_pairs[node] = pos_pairs
            
            neg_samples = random.sample(far_nodes, num_neg) if num_neg < len(far_nodes) else far_nodes
            neg_pairs = [(node, neg_node) for neg_node in neg_samples]
            self.negative_pairs.extend(neg_pairs)
            self.node_negative_pairs[node] = neg_pairs
             
    
    def get_neigh_weights(self,node,node_only=False):
        neighs = self.adj_lists[int(node)]
        weights = self.weight_lists[int(node)]
        if node_only:
            return neighs
        neigh_nodes, neigh_weights = [], []
        for i in range(len(neighs)):
            neigh_nodes.append(neighs[i])
            neigh_weights.append(weights[i])
        neigh_weights = special.softmax(neigh_weights)
        return neigh_nodes, neigh_weights    
    def get_positive_nodes(self, nodes, n_walks, walk_len):
        return self._run_random_walks(nodes, n_walks, walk_len)

    def _run_random_walks(self, nodes, n_walks, walk_len):
        for node in nodes:
            if len(self.adj_lists[int(node)])==0:
                self.node_positive_pairs[node] = []
                continue
            curr_pairs = []
            for i in range(n_walks):
                curr_node=node
                for j in range(walk_len):
                    neigh_nodes, neigh_weights = self.get_neigh_weights(curr_node)
                    next_node = np.random.choice(neigh_nodes, 1, p=neigh_weights)[0]
                    if next_node != node and next_node in self.sampled_nodes:
                        self.positive_pairs.append((node, next_node))
                        curr_pairs.append((node, next_node))
                    curr_node = next_node    
            self.node_positive_pairs[node] = curr_pairs
        return self.positive_pairs
    
    def get_negative_nodes(self, nodes,num_neg, neg_walk_len):
        for node in nodes:
            if node in self.far_node_cache:
                far_nodes=self.far_node_cache[node]
            else:
                neighbors, frontier = {node}, {node}
                for _ in range(neg_walk_len):
                    current=set()
                    for outer in frontier:
                        current |= set(self.get_neigh_weights(outer, node_only=True))
                    frontier = current - neighbors
                    neighbors |= current
                homo_nodes = self.get_homo_nodes(node)
                far_nodes = homo_nodes - neighbors
                self.far_node_cache[node] = far_nodes
                
            neg_samples = self.negative_sample(node, far_nodes, num_neg) if num_neg < len(far_nodes) else far_nodes
            self.negative_pairs.extend([(node, neg_node) for neg_node in neg_samples])
            self.node_negative_pairs[node] = [(node, neg_node) for neg_node in neg_samples]
        return self.negative_pairs    