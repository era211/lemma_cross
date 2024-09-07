import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pyhocon
import os
from inspect import getfullargspec

# relative path of config file


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias)


class CrossEncoder(nn.Module):
    def __init__(self, is_training=True, long=True, tokenizer=None, model=None,
                 linear_weights=None):
        super(CrossEncoder, self).__init__()
        self.tokenizer = tokenizer
        self.long = long

        if is_training:
            self.tokenizer.add_tokens(['<m>', '</m>'], special_tokens=True)
            self.tokenizer.add_tokens(['<doc-s>', '</doc-s>'], special_tokens=True)
            self.tokenizer.add_tokens(['<g>'], special_tokens=True)
            self.model = model
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.model = model

        self.start_id = self.tokenizer.encode('<m>', add_special_tokens=False)[0]  # 将<m>转换为token ID
        self.end_id = self.tokenizer.encode('</m>', add_special_tokens=False)[0]  # 将</m>转换为token ID

        self.hidden_size = self.model.config.hidden_size

        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        if linear_weights is None:
            self.linear.apply(init_weights)
        else:
            self.linear.load_state_dict(linear_weights)

    def generate_cls_arg_vectors(self, input_ids, attention_mask, position_ids,
                                 global_attention_mask, arg1, arg2):
        arg_names = set(getfullargspec(self.model).args)

        if self.long:
            # output = self.model(input_ids,
            #                     position_ids=position_ids,
            #                     attention_mask=attention_mask,
            #                     global_attention_mask=None)
            output = self.model(input_ids,
                                position_ids=position_ids,
                                attention_mask=attention_mask
                                )
        else:
            output = self.model(input_ids,
                                attention_mask=attention_mask)

        last_hidden_states = output.last_hidden_state  # (16,512,768)
        cls_vector = output.pooler_output  # (16,768)

        arg1_vec = None
        if arg1 is not None:  # arg1:(16,512)
            arg1_vec = (last_hidden_states * arg1.unsqueeze(-1)).sum(1)  # (16,768)  触发词的向量包含整个句子信息
        arg2_vec = None
        if arg2 is not None:
            arg2_vec = (last_hidden_states * arg2.unsqueeze(-1)).sum(1)

        return cls_vector, arg1_vec, arg2_vec

    def generate_model_output(self, input_ids, attention_mask, position_ids,
                              global_attention_mask, arg1, arg2):
        cls_vector, arg1_vec, arg2_vec = self.generate_cls_arg_vectors(input_ids, attention_mask, position_ids,
                                                                       global_attention_mask, arg1, arg2)

        return torch.cat([cls_vector, arg1_vec, arg2_vec, arg1_vec * arg2_vec], dim=1)

    def frozen_forward(self, input_):
        return self.linear(input_)

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                global_attention_mask=None, arg1=None, arg2=None, lm_only=False, pre_lm_out=False):

        # attention_mask[global_attention_mask == 1] = 2

        if pre_lm_out:
            return self.linear(input_ids)

        lm_output = self.generate_model_output(input_ids, attention_mask=attention_mask,
                                               global_attention_mask=global_attention_mask,
                                               position_ids=position_ids,
                                               arg1=arg1, arg2=arg2)
        if lm_only:
            return lm_output

        return self.linear(lm_output)


class COnlyCrossEncoder(nn.Module):
    def __init__(self, is_training=True, long=True, tokenizer=None, model=None,
                 linear_weights=None):
        super(COnlyCrossEncoder, self).__init__()
        self.tokenizer = tokenizer
        self.long = long

        if is_training:
            self.tokenizer.add_tokens(['<m>', '</m>'], special_tokens=True)
            self.tokenizer.add_tokens(['<doc-s>', '</doc-s>'], special_tokens=True)
            self.tokenizer.add_tokens(['<g>'], special_tokens=True)
            self.model = model
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.model = model

        self.start_id = self.tokenizer.encode('<m>', add_special_tokens=False)[0]  # 将<m>转换为token ID
        self.end_id = self.tokenizer.encode('</m>', add_special_tokens=False)[0]  # 将</m>转换为token ID

        self.hidden_size = self.model.config.hidden_size

        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        if linear_weights is None:
            self.linear.apply(init_weights)
        else:
            self.linear.load_state_dict(linear_weights)

    def generate_cls_arg_vectors(self, input_ids, attention_mask, position_ids,
                                 global_attention_mask):
        arg_names = set(getfullargspec(self.model).args)

        if self.long:
            # output = self.model(input_ids,
            #                     position_ids=position_ids,
            #                     attention_mask=attention_mask,
            #                     global_attention_mask=None)
            output = self.model(input_ids,
                                position_ids=position_ids,
                                attention_mask=attention_mask
                                )
        else:
            output = self.model(input_ids,
                                attention_mask=attention_mask)

        last_hidden_states = output.last_hidden_state  # (16,512,768)
        cls_vector = output.pooler_output  # (16,768)

        return cls_vector

    def generate_model_output(self, input_ids, attention_mask, position_ids,
                              global_attention_mask):
        cls_vector = self.generate_cls_arg_vectors(input_ids, attention_mask, position_ids,
                                                                       global_attention_mask)

        return torch.cat([cls_vector], dim=1)

    def frozen_forward(self, input_):
        return self.linear(input_)

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                global_attention_mask=None, lm_only=False, pre_lm_out=False):

        # attention_mask[global_attention_mask == 1] = 2

        if pre_lm_out:
            return self.linear(input_ids)

        lm_output = self.generate_model_output(input_ids, attention_mask=attention_mask,
                                               global_attention_mask=global_attention_mask,
                                               position_ids=position_ids)
        if lm_only:
            return lm_output

        return self.linear(lm_output)


class EOnlyCrossEncoder(nn.Module):
    def __init__(self, is_training=True, long=True, tokenizer=None, model=None,
                 linear_weights=None):
        super(EOnlyCrossEncoder, self).__init__()
        self.tokenizer = tokenizer
        self.long = long

        if is_training:
            self.tokenizer.add_tokens(['<m>', '</m>'], special_tokens=True)
            self.tokenizer.add_tokens(['<doc-s>', '</doc-s>'], special_tokens=True)
            self.tokenizer.add_tokens(['<g>'], special_tokens=True)
            self.model = model
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.model = model

        self.start_id = self.tokenizer.encode('<m>', add_special_tokens=False)[0]  # 将<m>转换为token ID
        self.end_id = self.tokenizer.encode('</m>', add_special_tokens=False)[0]  # 将</m>转换为token ID

        self.hidden_size = self.model.config.hidden_size

        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        if linear_weights is None:
            self.linear.apply(init_weights)
        else:
            self.linear.load_state_dict(linear_weights)

    def generate_cls_arg_vectors(self, input_ids, attention_mask, position_ids):
        arg_names = set(getfullargspec(self.model).args)

        if self.long:
            # output = self.model(input_ids,
            #                     position_ids=position_ids,
            #                     attention_mask=attention_mask,
            #                     global_attention_mask=None)
            output = self.model(input_ids,
                                position_ids=position_ids,
                                attention_mask=attention_mask
                                )
        else:
            output = self.model(input_ids,
                                attention_mask=attention_mask)

        last_hidden_states = output.last_hidden_state  # (16,512,768)
        cls_vector = output.pooler_output  # (16,768)

        # 假设 last_hidden_state 形状为 (batch_size, sequence_length, hidden_size)
        batch_size, seq_length, hidden_size = last_hidden_states.size()

        # 将 attention_mask 维度从 (batch_size, sequence_length) 扩展为 (batch_size, sequence_length, hidden_size)
        expanded_attention_mask = attention_mask.unsqueeze(-1).expand(batch_size, seq_length, hidden_size)

        # 将 attention_mask 应用到 last_hidden_state 上，过滤掉 padding 部分
        masked_hidden_state = last_hidden_states * expanded_attention_mask

        # 现在可以直接对触发词进行处理，提取有效 token 的隐藏层表示
        trigger_word1_hidden_states = masked_hidden_state[:, :256]  # 取第一个触发词的部分 (batch_size, 256, hidden_size)
        trigger_word2_hidden_states = masked_hidden_state[:, 256:]  # 取第二个触发词的部分 (batch_size, 256, hidden_size)

        # 计算每个触发词的平均表示
        trigger_word1_representations = trigger_word1_hidden_states.sum(dim=1) / attention_mask[:, :256].sum(dim=1, keepdim=True)
        trigger_word2_representations = trigger_word2_hidden_states.sum(dim=1) / attention_mask[:, 256:].sum(dim=1, keepdim=True)
        return cls_vector, trigger_word1_representations, trigger_word2_representations

    def generate_model_output(self, input_ids, attention_mask, position_ids):
        cls_vector, arg1_vec, arg2_vec = self.generate_cls_arg_vectors(input_ids, attention_mask, position_ids)

        return torch.cat([arg1_vec, arg2_vec, arg1_vec * arg2_vec], dim=1)

    def frozen_forward(self, input_):
        return self.linear(input_)

    def forward(self, input_ids, attention_mask=None, position_ids=None, lm_only=False, pre_lm_out=False):

        # attention_mask[global_attention_mask == 1] = 2

        if pre_lm_out:
            return self.linear(input_ids)

        lm_output = self.generate_model_output(input_ids, attention_mask=attention_mask,
                                               position_ids=position_ids)
        if lm_only:
            return lm_output

        return self.linear(lm_output)
