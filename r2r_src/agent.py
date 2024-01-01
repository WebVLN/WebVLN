# R2R-EnvDrop, 2019, haotan@cs.unc.edu
# Modified in Recurrent VLN-BERT, 2020, by Yicong.Hong@anu.edu.au

import json
import os
import sys
import numpy as np
import random
import math
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from env import R2RBatch
import utils
from utils import PAD, BOS, num_tokens, tok_bert, print_progress
import model_OSCAR, model_PREVALENT, QADecoder
import param
from param import args
from collections import defaultdict


class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(0)
        self.results = {}
        self.losses = [] # For learning agents

    def write_results(self):
        output = [{'idx':k, 'trajectory': v[0], 'answer': v[1]} for k,v in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def get_results(self):
        output = [{'idx': k, 'trajectory': v[0], 'answer': v[1]} for k, v in self.results.items()]
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=-1, **kwargs):
        print("Start Eval ...")
        # iters = 10 # for debug
        # self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        # eval_num = self.env.split_length+1
        eval_num = int((self.env.split_length-1)/4+1)*4
        # iters = 1000
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        # looped = False
        self.loss = 0
        if iters != -1:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['idx']] = [traj['path'], traj['answer']]
                print_progress(i+1, iters, prefix='Eval Progress:', suffix=' ', bar_length=50)
        else:   # Do a full round
            i = 0
            while i < eval_num:
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['idx']] = [traj['path'], traj['answer']]
                    print_progress(i+1, eval_num, prefix='Eval Progress:', suffix=' ', bar_length=50)
                    i +=1


class Seq2SeqAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    def __init__(self, env, results_path="", episode_len=20):
        super(Seq2SeqAgent, self).__init__(env, results_path)
        self.args = args

        self.episode_len = episode_len
        self.feature_size = self.env.feature_size

        # Models
        if args.vlnbert == 'oscar':
            self.vln_bert = model_OSCAR.VLNBERT(feature_size=self.feature_size).cuda()
            # self.critic = model_OSCAR.Critic().cuda()
        elif args.vlnbert == 'prevalent':
            self.vln_bert = model_PREVALENT.VLNBERT(feature_size=self.feature_size).cuda()
            # self.critic = model_PREVALENT.Critic().cuda()
        self.qa_decoder = QADecoder.QADecoder(args).cuda()

        self.models = (self.vln_bert, self.qa_decoder)

        # Optimizers
        self.vln_bert_optimizer = args.optimizer(self.vln_bert.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # self.critic_optimizer = args.optimizer(self.critic.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.qa_optimizer = args.optimizer(self.qa_decoder.parameters(), lr=20*args.lr, weight_decay=args.weight_decay)

        self.optimizers = (self.vln_bert_optimizer, self.qa_optimizer)


        # Evaluations
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid, size_average=False)
        self.qa_criterion = QADecoder.LabelSmoothingLoss(num_tokens, smoothing=0.1)
        # self.ndtw_criterion = utils.ndtw_initialize()

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

    def _sort_batch(self, obs):
        seq_tensor = np.array([ob['text_enc'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == PAD, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)  # True -> descending
        sorted_tensor = seq_tensor[perm_idx]
        mask = (sorted_tensor != PAD)

        token_type_ids = torch.zeros_like(mask)
# item['answer_enc'],
                # 'answer_enc_w_eos' : item['answer_enc_w_eos'],

        answer = torch.from_numpy(np.array([ob['answer_enc'] for ob in obs]))[perm_idx]
        answer_w_eos = torch.from_numpy(np.array([ob['answer_enc_w_eos'] for ob in obs]))[perm_idx]

        answer_unpad_positions = answer_w_eos != PAD

        return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
               mask.long().cuda(), token_type_ids.long().cuda(), \
               list(seq_lengths), list(perm_idx), \
               answer.long().cuda(), answer_w_eos.long().cuda(), answer_unpad_positions


    def _candidate_variable(self, obs):
        candidate_leng = [len(ob['candidate'])*3 + 1 for ob in obs]  # +1 is for the end
        candidate_feat = np.zeros((len(obs), max(candidate_leng), self.feature_size), dtype=np.float32)
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, cc in enumerate(ob['candidate']):
                for k, feat in enumerate(ob['candidate'][cc]["feature"]):
                    candidate_feat[i, j*3+k, :] = feat
        candidate_leng = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end

        return torch.from_numpy(candidate_feat).cuda(), candidate_leng

    def get_input_feat(self, obs):
        candidate_feat, candidate_leng = self._candidate_variable(obs)

        return candidate_feat, candidate_leng

    def _teacher_action(self, obs, ended, step):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            # a[i]=[]
            if ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid

            elif step == len(ob['gt_path'])-1:
                # a[i] = len(ob['candidate'])*3
                a[i] = len(ob['candidate'])

            elif step >= len(ob['gt_path']):
                a[i] = args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if ob['candidate'][candidate]['urlID'] == ob['gt_path'][step+1]:   # Next view point
                        # a[i] = k*3
                        a[i] = k
                        break
                # if ob['gt_path'][step][1] in ob['candidate']:
                #     for k, candidate in enumerate(ob['candidate']):
                #         if candidate == ob['gt_path'][step][1]:
                #             if ob['candidate'][candidate]['urlID'] == ob['gt_path'][step+1]:   # Next view point
                #                 # a[i] = k*3
                #                 a[i] = k
                #                 break
                #             else:
                #                 print("candidate != ob['gt_path'][step][1]")
                else:   # Stop here
                    print("Cur_choose not in cc!!! GAN!!! Sad!!!")
                    #a[i] = args.ignoreid # debuug
                    # assert ob['gt_path'][step] == ob['urlID']         # The teacher action should be "STAY HERE"
                    # a[i] = len(ob['candidate'])*3
                    # a[i].append(len(ob['candidate'])*3)
        return torch.from_numpy(a).cuda()

    def _argmax_action(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                          # Just ignore this index
                a[i] = args.ignoreid
            elif len(ob['teacher']) == 1: # Stop here
                a[i] = len(ob['candidate'])
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if ob['candidate'][candidate]['urlID'] == ob['teacher'][1]:   # Next view point
                        a[i] = k
                        break
                else:
                    print("No CC in cur_url")

        return torch.from_numpy(a).cuda()
    
    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        
        if perm_idx is None:
            perm_idx = range(len(perm_obs))

        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                cc_id = list(perm_obs[i]['candidate'].keys())[action]
                select_candidate = perm_obs[i]['candidate'][cc_id]
                src_point = perm_obs[i]['urlID']
                trg_point = select_candidate['urlID']
                self.env.env.sims[idx].makeAction(trg_point)
                if traj is not None:
                    traj[i]['path'].append(trg_point)

    def rollout(self, train_ml=None, train_rl=True, reset=True, qa_w=None):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment

        :return:
        """
        # if self.feedback == 'teacher' or self.feedback == 'argmax':
        #     train_rl = False

        if reset:  # Reset env
            obs = np.array(self.env.reset())
        else:
            obs = np.array(self.env._get_obs())

        batch_size = len(obs)

        # Language input
        sentence, language_attention_mask, token_type_ids, seq_lengths, perm_idx, answer, answer_w_eos, answer_unpad_positions = self._sort_batch(obs)
        perm_obs = obs[perm_idx]

        ''' Language BERT '''
        language_inputs = {'mode':        'language',
                        'sentence':       sentence,
                        'attention_mask': language_attention_mask,
                        'lang_mask':      language_attention_mask,
                        'token_type_ids': token_type_ids}
        if args.vlnbert == 'oscar':
            language_features = self.vln_bert(**language_inputs)
        elif args.vlnbert == 'prevalent':
            h_t, language_features = self.vln_bert(**language_inputs)

        # Record starting point
        traj = [{
            'idx': ob['idx'],
            'path': [ob['urlID']],
            'answer': [''],
        } for ob in perm_obs]

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        # last_ndtw = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(perm_obs):   # The init distance from the view point to the target
            last_dist[i] = ob['distance']
            path_act = [vp for vp in traj[i]['path']]
            # last_ndtw[i] = self.ndtw_criterion[ob['scan']](path_act, ob['gt_path'], metric='ndtw')

        # Initialization the tracking state
        ended = np.array([False] * batch_size)  # Indices match permuation of the model, not env

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        ml_loss = 0.
        qa_emb=[]

        for t in range(self.episode_len):
            candidate_feat, candidate_leng = self.get_input_feat(perm_obs)

            # the first [CLS] token, initialized by the language BERT, serves
            # as the agent's state passing through time steps
            if (t >= 1) or (args.vlnbert=='prevalent'):
                language_features = torch.cat((h_t.unsqueeze(1), language_features[:,1:,:]), dim=1)

            visual_temp_mask = (utils.length2mask(candidate_leng) == 0).long()
            visual_attention_mask = torch.cat((language_attention_mask, visual_temp_mask), dim=-1)

            self.vln_bert.vln_bert.config.directions = max(candidate_leng)
            ''' Visual BERT '''
            visual_inputs = {'mode':              'visual',
                            'sentence':           language_features,
                            'attention_mask':     visual_attention_mask,
                            'lang_mask':          language_attention_mask,
                            'vis_mask':           visual_temp_mask,
                            'token_type_ids':     token_type_ids,
                            'cand_feats':         candidate_feat}
            h_t, logit = self.vln_bert(**visual_inputs)
            hidden_states.append(h_t)

            # Mask outputs where agent can't move forward
            # Here the logit is [b, max_candidate]
            candidate_mask = utils.length2mask(candidate_leng)
            logit.masked_fill_(candidate_mask, -float('inf'))

            target = self._argmax_action(perm_obs, ended)
            ml_loss += self.criterion(logit, target)
            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                 # teacher forcing
            elif self.feedback == 'argmax':
                # a_t = logit[:,::3]
                # a_t = a_t.detach()
                # _, a_t = a_t.max(1)        # student forcing - argmax
                # a_t = a_t*3
                # log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
                _, a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
                # log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
                # policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))   # Gather the log_prob for each batch
            elif self.feedback == 'sample':
                probs = F.softmax(logit, 1)  # sampling an action from model
                c = torch.distributions.Categorical(probs)
                # self.logs['entropy'].append(c.entropy().sum().item())            # For log
                # entropys.append(c.entropy())                                     # For optimization
                a_t = c.sample().detach()
                # policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')
            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (candidate_leng[i]-1) or next_id == args.ignoreid or ended[i]:    # The last action is <end>
                    cpu_a_t[i] = -1             # Change the <end> and ignore action to -1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)
            obs = np.array(self.env._get_obs())
            perm_obs = obs[perm_idx]            # Perm the obs for the resu

            # if train_rl:
            #     # Calculate the mask and reward
            #     dist = np.zeros(batch_size, np.float32)
            #     ndtw_score = np.zeros(batch_size, np.float32)
            #     reward = np.zeros(batch_size, np.float32)
            #     mask = np.ones(batch_size, np.float32)
            #     for i, ob in enumerate(perm_obs):
            #         dist[i] = ob['distance']
            #         path_act = [vp[0] for vp in traj[i]['path']]
            #         ndtw_score[i] = self.ndtw_criterion[ob['scan']](path_act, ob['gt_path'], metric='ndtw')

            #         if ended[i]:
            #             reward[i] = 0.0
            #             mask[i] = 0.0
            #         else:
            #             action_idx = cpu_a_t[i]
            #             # Target reward
            #             if action_idx == -1:                              # If the action now is end
            #                 if dist[i] < 3.0:                             # Correct
            #                     reward[i] = 2.0 + ndtw_score[i] * 2.0
            #                 else:                                         # Incorrect
            #                     reward[i] = -2.0
            #             else:                                             # The action is not end
            #                 # Path fidelity rewards (distance & nDTW)
            #                 reward[i] = - (dist[i] - last_dist[i])
            #                 ndtw_reward = ndtw_score[i] - last_ndtw[i]
            #                 if reward[i] > 0.0:                           # Quantification
            #                     reward[i] = 1.0 + ndtw_reward
            #                 elif reward[i] < 0.0:
            #                     reward[i] = -1.0 + ndtw_reward
            #                 else:
            #                     raise NameError("The action doesn't change the move")
            #                 # Miss the target penalty
            #                 if (last_dist[i] <= 1.0) and (dist[i]-last_dist[i] > 0.0):
            #                     reward[i] -= (1.0 - last_dist[i]) * 2.0
            #     rewards.append(reward)
            #     masks.append(mask)
            #     last_dist[:] = dist
            #     last_ndtw[:] = ndtw_score

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break




        # if train_rl:
        #     # Last action in A2C
        #     input_a_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)

        #     language_features = torch.cat((h_t.unsqueeze(1), language_features[:,1:,:]), dim=1)

        #     visual_temp_mask = (utils.length2mask(candidate_leng) == 0).long()
        #     visual_attention_mask = torch.cat((language_attention_mask, visual_temp_mask), dim=-1)

        #     self.vln_bert.vln_bert.config.directions = max(candidate_leng)
        #     ''' Visual BERT '''
        #     visual_inputs = {'mode':              'visual',
        #                     'sentence':           language_features,
        #                     'attention_mask':     visual_attention_mask,
        #                     'lang_mask':          language_attention_mask,
        #                     'vis_mask':           visual_temp_mask,
        #                     'token_type_ids':     token_type_ids,
        #                     'action_feats':       input_a_t,
        #                     # 'pano_feats':         f_t,
        #                     'cand_feats':         candidate_feat}
        #     last_h_, _ = self.vln_bert(**visual_inputs)

        #     rl_loss = 0.

        #     # NOW, A2C!!!
        #     # Calculate the final discounted reward
        #     last_value__ = self.critic(last_h_).detach()        # The value esti of the last state, remove the grad for safety
        #     discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
        #     for i in range(batch_size):
        #         if not ended[i]:        # If the action is not ended, use the value function as the last reward
        #             discount_reward[i] = last_value__[i]

        #     length = len(rewards)
        #     total = 0
        #     for t in range(length-1, -1, -1):
        #         discount_reward = discount_reward * args.gamma + rewards[t]  # If it ended, the reward will be 0
        #         mask_ = Variable(torch.from_numpy(masks[t]), requires_grad=False).cuda()
        #         clip_reward = discount_reward.copy()
        #         r_ = Variable(torch.from_numpy(clip_reward), requires_grad=False).cuda()
        #         v_ = self.critic(hidden_states[t])
        #         a_ = (r_ - v_).detach()

        #         rl_loss += (-policy_log_probs[t] * a_ * mask_).sum()
        #         rl_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5  # 1/2 L2 loss
        #         if self.feedback == 'sample':
        #             rl_loss += (- 0.01 * entropys[t] * mask_).sum()
        #         self.logs['critic_loss'].append((((r_ - v_) ** 2) * mask_).sum().item())

        #         total = total + np.sum(masks[t])
        #     self.logs['total'].append(total)

        #     # Normalize the loss function
        #     if args.normalize_loss == 'total':
        #         rl_loss /= total
        #     elif args.normalize_loss == 'batch':
        #         rl_loss /= batch_size
        #     else:
        #         assert args.normalize_loss == 'none'

        #     self.loss += rl_loss
        #     self.logs['RL_loss'].append(rl_loss.item())

        # if train_ml is not None:
        #     # ml_loss = ml_loss * train_ml / batch_size
        #     ml_loss = ml_loss / batch_size
        #     self.loss += ml_loss
        #     self.logs['IL_loss'].append(ml_loss.item())
        # else:
        #     ml_loss = ml_loss / batch_size
        #     self.loss += ml_loss
        #     self.logs['val_IL_loss'].append(ml_loss.item())
        if train_ml is not None:
            ml_loss = ml_loss * train_ml / batch_size
        else:
            ml_loss = ml_loss / batch_size
        self.loss += ml_loss
        self.logs['IL_loss'].append(ml_loss.item())

        for b_i, traj_i in enumerate(traj):
            cur_idx = len(traj_i["path"])-2
            qa_emb.append(hidden_states[cur_idx][b_i].unsqueeze(0))
        qa_emb = torch.stack(qa_emb)

        if self.feedback == 'argmax':
            current_token_ids = sentence.new_full([batch_size, 1], BOS)
            for _ in range(40):
                preds = self.qa_decoder(qa_emb, current_token_ids, eval_flag=True)
                next_token_id = preds.max(dim=2)[1]
                current_token_ids = torch.cat([current_token_ids, next_token_id], dim=1)
                for i, caption in enumerate(current_token_ids):
                    caption = tok_bert.decode(caption.cpu().numpy(), skip_special_tokens=False)
                    caption = caption.split('[unused3]')[0][9:].strip()
                    traj[i]["answer"] = caption
                    # if batch[6][i] not in captions.keys():
                    #     captions[batch[6][i]] = [{'caption': caption, 
                    #                             'mode': int(mode_id.cpu())}]
                    # else:
                    #     captions[batch[6][i]].append({'caption': caption, 
                    #                                 'mode': int(mode_id.cpu())})
        else:
            pred_logits = self.qa_decoder(qa_emb, answer)
            pred_logits = pred_logits[answer_unpad_positions]
            answer_w_eos_ = answer_w_eos[answer_unpad_positions]
            
            qa_loss = self.qa_criterion(pred_logits, answer_w_eos_)
            if qa_w is not None:
                qa_loss = qa_loss*qa_w
            self.loss += qa_loss
            self.logs['IL_loss'].append(ml_loss.item())
            self.logs['QA_loss'].append(qa_loss.item())

        return traj

    def test(self, use_dropout=False, feedback='argmax', iters=-1):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        # self.feedback = 'teacher'
        if use_dropout:
            self.vln_bert.train()
            # self.critic.train()
            self.qa_decoder.train()
        else:
            self.vln_bert.eval()
            # self.critic.eval()
            self.qa_decoder.eval()
        super(Seq2SeqAgent, self).test(iters)

    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

    # def accumulate_gradient(self, feedback='teacher', **kwargs):
    #     if feedback == 'teacher':
    #         self.feedback = 'teacher'
    #         self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
    #     elif feedback == 'sample':
    #         self.feedback = 'teacher'
    #         self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
    #         self.feedback = 'sample'
    #         self.rollout(train_ml=None, train_rl=True, **kwargs)
    #     else:
    #         assert False

    def optim_step(self):
        self.loss.backward()

        torch.nn.utils.clip_grad_norm(self.vln_bert.parameters(), 40.)

        self.vln_bert_optimizer.step()
        # self.critic_optimizer.step()
        self.qa_optimizer.step()

    def train(self, n_iters, feedback='teacher', writer=None, idx=None, **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback

        self.vln_bert.train()
        # self.critic.train()
        self.qa_decoder.train()

        self.losses = []
        for i_ in range(0, n_iters):

            self.vln_bert_optimizer.zero_grad()
            # self.critic_optimizer.zero_grad()
            self.qa_optimizer.zero_grad()

            self.loss = 0

            if feedback == 'teacher':
                self.rollout(train_ml=args.teacher_weight, train_rl=False, qa_w=args.qa_w, **kwargs)
            elif feedback == 'argmax':
                self.rollout(train_ml=args.teacher_weight, train_rl=False, qa_w=args.qa_w, **kwargs)
            elif feedback == 'sample':
                self.rollout(train_ml=args.teacher_weight, train_rl=False, qa_w=args.qa_w, **kwargs)
            elif feedback == 'mix':  # agents in IL and RL separately
                self.feedback = 'sample'
                self.rollout(train_ml=1.0, train_rl=True, qa_w=args.qa_w, **kwargs)
                self.feedback = 'teacher'
                self.rollout(train_ml=args.teacher_weight, train_rl=True, qa_w=args.qa_w, **kwargs)
                # if args.ml_weight != 0:
                #     self.feedback = 'teacher'
                #     self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
                # self.feedback = 'sample'
                # self.rollout(train_ml=None, train_rl=True, **kwargs)
            else:
                assert False

            self.loss.backward()

            torch.nn.utils.clip_grad_norm(self.vln_bert.parameters(), 40.)

            self.vln_bert_optimizer.step()
            # self.critic_optimizer.step()
            self.qa_optimizer.step()

            print_progress(i_+1, n_iters, prefix='Training Progress:', suffix='Complete', bar_length=50)
            if (i_+1)%100==0:
                IL_loss = sum(self.logs['IL_loss']) / max(len(self.logs['IL_loss']), 1)
                writer.add_scalar("loss/train_IL_loss", IL_loss, idx+i_+1)
                QA_loss = sum(self.logs['QA_loss']) / max(len(self.logs['QA_loss']), 1)
                writer.add_scalar("loss/train_QA_loss", QA_loss, idx+i_+1)
                self.logs = defaultdict(list)


    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                    #  ("critic", self.critic, self.critic_optimizer),
                     ("qa_decoder", self.qa_decoder, self.qa_optimizer),]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            # model_keys = set(state.keys())
            # load_keys = set(states[name]['state_dict'].keys())
            # if model_keys != load_keys:
            #     print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            if args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                    #  ("critic", self.critic, self.critic_optimizer),
                     ("qa_decoder", self.qa_decoder, self.qa_optimizer),]
        for param in all_tuple:
            recover_state(*param)
        return states['vln_bert']['epoch'] - 1
