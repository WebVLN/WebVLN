''' Batched Room-to-Room navigation environment '''

import sys
import csv
import numpy as np
import math
import base64
import utils
import json
import os
import random
random.seed(0)
from param import args
import pickle as pkl

from vlnbert.vlnbert_init import get_tokenizer

csv.field_size_limit(sys.maxsize)

class Simulator(object):
    '''A simple simulator for different websites'''

    def __init__(
            self,
            shortest_paths,
            connectivity
        ) -> None:
        self.websiteID = ''
        self.urlID = ''
        self.shortest_paths = shortest_paths
        self.connectivity = connectivity
    
    def load_candidate(
            self,
            websiteID: str,
            urlID: str,
        ) -> dict:
        ''' Load the candidate from the database '''
        # TODO: load the candidate from the database
        return self.connectivity[websiteID][urlID]["data"]

    def newEpisode(
            self,
            websiteID: str,
            urlID: str,) -> None:
        self.urlID = urlID
        self.websiteID = websiteID
        # load the candidate
        self.candidate = self.load_candidate(websiteID, urlID)
    
    def getState(self):
        ''' Get the current state of the simulator '''
        self.state = {
            "websiteID": self.websiteID,
            "urlID": self.urlID,
            "candidate": self.candidate,
        }
        return self.state
    
    def makeAction(
            self,
            urlID: str,) -> None:
        self.urlID = urlID
        # load the candidate
        self.candidate = self.load_candidate(self.websiteID, urlID)


class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''

    def __init__(self, feat_dict, shortest_paths, connectivity, batch_size=16):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feature_store: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        self.feature_size=512
        self.features = feat_dict
        self.sims = []
        for i in range(batch_size):
            sim = Simulator(shortest_paths, connectivity)
            self.sims.append(sim)

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def newEpisodes(self, websiteID, urlID):
        for i, (websiteID, urlID) in enumerate(zip(websiteID, urlID)):
            self.sims[i].newEpisode(websiteID, urlID)

    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((30, 2048), sim_state) ] * batch_size
        """
        states=[]
        for i, sim in enumerate(self.sims):
            state = sim.getState()
            # Get candidate
            candidate = state["candidate"]
            # Load the feature
            feats=[]
            for clickable_id in candidate:
                if len(candidate[clickable_id]["imgs"]) > 0:
                    feats.append(self.features[candidate[clickable_id]["imgs"][0]])
                else:
                    feats.append(np.zeros((1,512)))
            states.append([feats, state])


        return states

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. '''
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction(index, heading, elevation)


class R2RBatch():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, 
            feature_store, text_features, screenshot_features, data_dir, batch_size=100, seed=0, splits=['train'], data=None,
            name=None,
            allocated_episodes=None,
        ):
        self.shortest_paths = json.load(open(f"{data_dir}/shortest_paths.json", 'r'))
        self.connectivity = json.load(open(f"{data_dir}/map.json", 'r'))
        self.text_features = text_features
        self.screenshot_features = screenshot_features
        self.env = EnvBatch(feature_store, self.shortest_paths, self.connectivity, batch_size)
        if feature_store:
            self.feature_size = self.env.feature_size
        else:
            self.feature_size = 512
        self.data = data
        self.scans = set([x['path'][0].split("_")[0] for x in self.data])

        if name is None:
            self.name = splits[0] if len(splits) > 0 else "FAKE"
        else:
            self.name = name

        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        if self.splits[0] == "train":
            random.shuffle(self.data)
        self.split_length = len(self.data)

        self.ix = 0
        self.batch_size = batch_size

        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

    def size(self):
        return len(self.data)


    def _next_minibatch(self, tile_one=False, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        if batch_size is None:
            batch_size = self.batch_size
        if tile_one:
            batch = [self.data[self.ix]] * batch_size
            self.ix += 1
            if self.ix >= len(self.data):
                random.shuffle(self.data)
                self.ix -= len(self.data)
        else:
            batch = self.data[self.ix: self.ix+batch_size]
            if len(batch) < batch_size:
                if self.splits == "train":
                    random.shuffle(self.data)
                self.ix = batch_size - len(batch)
                batch += self.data[:self.ix]
            else:
                self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0



    
    def make_candidate(self, feature, state):
        candidate = {}
        for idx, ckable_id_id in enumerate(state['candidate']):
            cur_cc = state['candidate'][ckable_id_id]
            # tok_bert = get_tokenizer(args)
            # ''' BERT tokenizer '''
            # if len(cur_cc['text'])>0:
            #     text = f"Going to: {cur_cc['href_full']}, description: {cur_cc['text'][0]}"
            # else:
            #     text = f"Going to: {cur_cc['href_full']}"
            # instr_tokens = tok_bert.tokenize(text)
            # padded_instr_tokens, num_words = pad_instr_tokens(instr_tokens, args.maxInput)
            # text_enc = tok_bert.convert_tokens_to_ids(padded_instr_tokens)

            
            if cur_cc["clickable_id"] in self.text_features:
                # candidate[ckable_id_id] = [self.text_features[cur_cc["clickable_id"]], feature[idx], pos_feature]
                candidate[ckable_id_id] = {
                            'urlID': cur_cc["next_url_id"],
                            'feature': [self.text_features[cur_cc["clickable_id"]], feature[idx], self.screenshot_features[cur_cc["clickable_id"]]]
                        }
            else:
                # candidate[ckable_id_id] = [np.zeros((1,512)), feature[idx], pos_feature]
                candidate[ckable_id_id] = {
                            'urlID': cur_cc["next_url_id"],
                            'feature': [np.zeros((1,512)), feature[idx], self.screenshot_features[cur_cc["clickable_id"]]]
                        }
                

        return candidate

    def _get_obs(self):
        # prepare train data
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            

            # if feature is None:
            #     feature = np.zeros((64, 512))

            # Full features
            candidate_feature = self.make_candidate(feature, state)
            # [visual_feature, angle_feature] for views
            # feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)

            obs.append(
                {
                'idx' : item['idx'],
                'websiteID' : state['websiteID'],
                'urlID' : state['urlID'],
                'candidate' : candidate_feature,
                'text' : item['text'],
                'text_enc' : item['text_enc'],
                'teacher' : self.shortest_paths[state['websiteID']][state['urlID']][item['path'][-1]],
                'gt_path' : item['path'],
                'distance' : len(self.shortest_paths[state['websiteID']][state['urlID']][item['path'][-1]]),
                'answer' : item['A'],
                'answer_enc' : item['answer_enc'],
                'answer_enc_w_eos' : item['answer_enc_w_eos'],
                # 'text_enc' : item['text_enc'],
            })
            # if 'text_enc' in item:
            #     obs[-1]['text_enc'] = item['text_enc']
            # A2C reward. The negative distance between the state and the final state
            # obs[-1]['distance'] = len(self.connectivity[state['urlID']][item['path'][-1]])
        return obs

    def reset(self, batch=None, inject=False, **kwargs):
        ''' Load a new minibatch / episodes. '''
        if batch is None:       # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        else:
            if inject:          # Inject the batch into the next minibatch
                self._next_minibatch(**kwargs)
                self.batch[:len(batch)] = batch
            else:               # Else set the batch to the current batch
                self.batch = batch
        scanIds = [item['path'][0].split("_")[0] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds)
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()

    # def get_statistics(self):
    #     stats = {}
    #     length = 0
    #     path = 0
    #     for datum in self.data:
    #         length += len(self.tok.split_sentence(datum['instructions']))
    #         path += self.distances[datum['scan']][datum['path'][0]][datum['path'][-1]]
    #     stats['length'] = length / len(self.data)
    #     stats['path'] = path / len(self.data)
    #     return stats
