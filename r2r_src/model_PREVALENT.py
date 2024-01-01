# Recurrent VLN-BERT, 2020, by Yicong.Hong@anu.edu.au

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from param import args

from vlnbert.vlnbert_init import get_vlnbert_models

class VLNBERT(nn.Module):
    def __init__(self, feature_size=512):
        super(VLNBERT, self).__init__()
        print('\nInitalizing the VLN-BERT model ...')

        self.vln_bert = get_vlnbert_models(args)  # initialize the VLN-BERT
        self.vln_bert.config.directions = 4  # a preset random number

        hidden_size = self.vln_bert.config.hidden_size
        layer_norm_eps = self.vln_bert.config.layer_norm_eps

        # self.action_state_project = nn.Sequential(
        #     nn.Linear(hidden_size*3, hidden_size), nn.Tanh())
        # self.action_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)
        self.cc_project = nn.Sequential(
            nn.Linear(512*3, 512), nn.ReLU())
        self.cc_LayerNorm = BertLayerNorm(512, eps=layer_norm_eps)

        # self.cc_project_1 = nn.Sequential(
        #     nn.Linear(512, hidden_size), nn.ReLU())
        # self.cc_LayerNorm_1 = BertLayerNorm(hidden_size, eps=layer_norm_eps)

        self.drop_env = nn.Dropout(p=args.featdropout)
        self.img_projection = nn.Linear(feature_size, hidden_size, bias=True)
        self.cand_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)

        self.vis_lang_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)
        self.state_proj = nn.Linear(hidden_size*2, hidden_size, bias=True)
        self.state_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, mode, sentence, token_type_ids=None,
                attention_mask=None, lang_mask=None, vis_mask=None,
                position_ids=None, action_feats=None, pano_feats=None, cand_feats=None):

        if mode == 'language':
            init_state, encoded_sentence = self.vln_bert(mode, sentence, attention_mask=attention_mask, lang_mask=lang_mask,)
            return init_state, encoded_sentence

        elif mode == 'visual':
            batch_size,_,h_dim = cand_feats.shape
            cand_feats_cc = cand_feats[:,:-1,:]
            cand_feats_cc = self.drop_env(cand_feats_cc)
            cand_feats_cc = cand_feats_cc.reshape(batch_size,-1,h_dim*3)
            cand_feats_cc = self.cc_project(cand_feats_cc)
            cand_feats_cc = self.cc_LayerNorm(cand_feats_cc)
            cand_feats_new = torch.cat((cand_feats_cc, cand_feats[:,-1:,:]), dim=1)

            h_t, logit, attended_language, attended_visual = self.vln_bert(mode, sentence, attention_mask=attention_mask, lang_mask=lang_mask, vis_mask=vis_mask, img_feats=cand_feats_new)

            # update agent's state, unify history, language and vision by elementwise product
            vis_lang_feat = self.vis_lang_LayerNorm(attended_language * attended_visual)
            state_output = torch.cat((h_t, vis_lang_feat), dim=-1)
            state_proj = self.state_proj(state_output)
            state_proj = self.state_LayerNorm(state_proj)

            return state_proj, logit

        else:
            ModuleNotFoundError


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


# class Critic(nn.Module):
#     def __init__(self):
#         super(Critic, self).__init__()
#         self.state2value = nn.Sequential(
#             nn.Linear(768, 512),
#             nn.ReLU(),
#             nn.Dropout(args.dropout),
#             nn.Linear(512, 1),
#         )

#     def forward(self, state):
#         return self.state2value(state).squeeze()
