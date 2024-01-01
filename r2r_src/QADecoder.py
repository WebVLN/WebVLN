import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
from transformers.models.bert.modeling_bert import (BertEncoder, 
                                                    BertPredictionHeadTransform,
                                                    BertLMPredictionHead,
                                                    BertEmbeddings)
from transformers.models.bert.configuration_bert import BertConfig

from utils import PAD, BOS, tok_bert

class Embeddings(nn.Module):
    def __init__(self, bert_cfg):
        super().__init__()
        self.word_embeddings = nn.Embedding(bert_cfg.vocab_size, 
                                            bert_cfg.hidden_size, 
                                            padding_idx=PAD)
        self.position_embeddings = nn.Embedding(768, bert_cfg.hidden_size)
        self.LayerNorm = nn.LayerNorm(bert_cfg.hidden_size,
                                       eps=bert_cfg.layer_norm_eps)
        self.dropout = nn.Dropout(bert_cfg.hidden_dropout_prob)

        self.register_buffer(
            "position_ids", 
            torch.arange(bert_cfg.max_position_embeddings).expand((1, -1))
        )

    def forward(self, input_ids):
        position_ids = self.position_ids[:, :input_ids.size(1)]
        position_embeds = self.position_embeddings(position_ids)
        inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
class QADecoder(nn.Module):
    def __init__(self, args, num_layers=2):
        super().__init__()
        bert_cfg = BertConfig(
            hidden_size=768,
            hidden_dropout_prob=args.dropout,
            num_hidden_layers=num_layers,
            is_decoder=True,
            add_cross_attention=True,
        )
        self.embedding_layer = Embeddings(bert_cfg)
        self.encoder = BertEncoder(bert_cfg)
        self.classifier = BertLMPredictionHead(bert_cfg)

        position_ids = torch.arange(bert_cfg.max_position_embeddings)
        attn_mask = position_ids[None, None, :].repeat(
            1, bert_cfg.max_position_embeddings, 1
        ) <= position_ids[None, :, None]
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, img_feat, token_ids, eval_flag=False):
        batch_size = token_ids.size(0) # [n,l]

        # start_token = token_ids.new_full([batch_size, 1], BOS)
        # token_ids = torch.cat([start_token, token_ids], dim=1)
        embeds = self.embedding_layer(token_ids)

        seq_len = token_ids.size(1)
        attn_mask = self.attn_mask[:, :seq_len, :seq_len]
        attn_mask = attn_mask.repeat(batch_size, 1, 1)
        attn_mask = attn_mask[:, None, :, :].float()
        attn_mask = (1.0 - attn_mask) * -10000.0

        hidden_state = self.encoder(embeds, 
                                    attention_mask=attn_mask, 
                                    encoder_hidden_states=img_feat)
        if eval_flag:
            preds = self.classifier(hidden_state.last_hidden_state[:, [-1], :])
        else:
            preds = self.classifier(hidden_state.last_hidden_state)
        return preds
    

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, weight=None, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        self.weight = weight

    def forward(self, pred, target):
        """
        Args:
            pred: (N, C), float
            target: (N,), long, values in [0, C-1]
        """
        if self.weight is None:
            self.weight = torch.ones(self.classes, dtype=torch.float32,
                                     device=target.device)
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        weight = self.weight[target]
        weighted_loss = torch.sum(-true_dist * pred, dim=-1) * weight

        return torch.mean(weighted_loss) * weight.numel() / weight.sum()
