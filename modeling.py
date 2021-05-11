# coding=utf-8

import os
import pdb
import torch
import torch.nn.functional as F
from torch import nn

from torch.nn import CrossEntropyLoss
from transformers import (
        BertConfig,
        BertModel,
        BertForTokenClassification,
        BertTokenizer,
        RobertaConfig,
        RobertaForTokenClassification,
        RobertaTokenizer
)

from torchcrf import CRF

class NER(BertForTokenClassification):
    def __init__(self, config, num_labels=3, random_bias=False, freq_bias=False, pmi_bias=True, mixin_bias=False, penalty=False, lambda_val=0.03, length_adaptive=True):
        super(NER, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)
        self.random_bias = random_bias
        self.freq_bias = freq_bias
        self.pmi_bias = pmi_bias 
        self.mixin_bias = mixin_bias
        if self.mixin_bias:
            self.mixin = nn.Linear(config.hidden_size, 1, bias=False)
        self.penalty = penalty
        self.lambda_val = lambda_val
        self.length_adaptive = length_adaptive
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, bias_tensor=None, data_type=None, temp_ids=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)

        # below process is only used at training time
        if data_type[0][0].item() == 1:
            if self.random_bias:
                # code of Bias Product method with random value
                rand_logits = torch.rand(batch_size, max_len, self.num_labels).cuda()
            elif self.freq_bias or self.pmi_bias:
                if self.length_adaptive:
                    # length adaptive
                    length_adaptive_temp_ids = torch.as_tensor([[1 + self.lambda_val * i if i > 1 else i for _, i in enumerate(temp_ids[j].tolist())] for j in range(len(temp_ids))]).to(temp_ids.device)
                    length_adaptive_temp_ids = length_adaptive_temp_ids.unsqueeze(2).expand(bias_tensor.size(0), bias_tensor.size(1), bias_tensor.size(2))
                    
                if self.mixin_bias:
                    # code from ko et al., 2020: https://github.com/dmis-lab/position-bias/blob/master/models/pytorch_bert/modeling_bert.py
                    mixin_coef = self.mixin(sequence_output)
                    mixin_coef = torch.max(mixin_coef, 1).values
                    mixin_coef = F.softplus(mixin_coef)
                    mixin_coef = mixin_coef.unsqueeze(2).expand(bias_tensor.size(0), bias_tensor.size(1), bias_tensor.size(2))
                    if self.length_adaptive:
                        bias_tensor = mixin_coef * bias_tensor / length_adaptive_temp_ids
                    else:
                        bias_tensor = mixin_coef * bias_tensor
                else:
                    # code of Bias Product method with statistics
                    # temperature scaling, guo et al., 2017
                    if self.length_adaptive:
                        bias_tensor = bias_tensor / length_adaptive_temp_ids
                    else:
                        pass

            else:
                pass

        outputs = (logits, sequence_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:

                if self.random_bias:
                    logits = logits + rand_logits
                elif self.freq_bias or self.pmi_bias:
                    logits = logits + bias_tensor

                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
                if self.penalty:
                    bias_lp = F.log_softmax(bias_tensor, dim=2)
                    entropy = -1 * torch.mean(torch.sum(torch.exp(bias_lp) * bias_lp, -1))
                    loss += entropy

                return ((loss,) + outputs)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                return loss
        else:
            return logits
