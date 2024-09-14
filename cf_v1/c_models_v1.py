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
    def __init__(self, is_training=True, long=True, tokenizer=None, model_name=None,
                 linear_weights=None, c_only_linear_weights=None, e_only_linear_weights=None):
        super(CrossEncoder, self).__init__()
        self.tokenizer = tokenizer
        self.long = long

        if is_training:
            self.tokenizer.add_tokens(['<m>', '</m>'], special_tokens=True)
            self.tokenizer.add_tokens(['<doc-s>', '</doc-s>'], special_tokens=True)
            self.tokenizer.add_tokens(['<g>'], special_tokens=True)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

        self.start_id = self.tokenizer.encode('<m>', add_special_tokens=False)[0]  # 将<m>转换为token ID
        self.end_id = self.tokenizer.encode('</m>', add_special_tokens=False)[0]  # 将</m>转换为token ID

        self.hidden_size = self.model.config.hidden_size

        # ff
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

        # c only
        self.c_linear = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        if c_only_linear_weights is None:
            self.c_linear.apply(init_weights)
        else:
            self.c_linear.load_state_dict(c_only_linear_weights)

        # e only
        self.e_linear = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        if e_only_linear_weights is None:
            self.e_linear.apply(init_weights)
        else:
            self.e_linear.load_state_dict(e_only_linear_weights)

    def generate_cls_arg_vectors(self, input_ids, attention_mask, position_ids,
                                 global_attention_mask=None, arg1=None, arg2=None, args_ff=True, args_cof=False, args_eof=False):
        arg_names = set(getfullargspec(self.model).args)

        if args_ff:
            if self.long:
                output = self.model(input_ids,
                                     position_ids=position_ids,
                                     attention_mask=attention_mask,
                #                     global_attention_mask=None
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

        if args_cof:
            if self.long:
                output = self.model(input_ids,
                                    position_ids=position_ids,
                                    attention_mask=attention_mask,
                                    # global_attention_mask=None
                                    )
            else:
                output = self.model(input_ids,
                                    attention_mask=attention_mask)

            last_hidden_states = output.last_hidden_state  # (16,512,768)
            cls_vector = output.pooler_output  # (16,768)

            return cls_vector

        if args_eof:
            if self.long:
                output = self.model(input_ids,
                                    position_ids=position_ids,
                                    attention_mask=attention_mask,
                                    # global_attention_mask=None
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

    def generate_model_output(self, f_input_ids, f_attention_mask, f_position_ids, f_global_attention_mask, f_arg1, f_arg2,
                              co_input_ids, co_attention_mask, co_position_ids, co_global_attention_mask,
                              eo_input_ids, eo_attention_mask, eo_position_ids):

        f_cls_vector, f_arg1_vec, f_arg2_vec = self.generate_cls_arg_vectors(f_input_ids, f_attention_mask, f_position_ids,
                                                                            f_global_attention_mask, f_arg1, f_arg2,
                                                                            args_ff=True, args_cof=False, args_eof=False)

        co_cls_vector = self.generate_cls_arg_vectors(co_input_ids, co_attention_mask, co_position_ids,
                                                        co_global_attention_mask,
                                                        args_ff=False, args_cof=True, args_eof=False)

        eo_cls_vector, eo_arg1_vec, eo_arg2_vec = self.generate_cls_arg_vectors(eo_input_ids, eo_attention_mask, eo_position_ids,
                                                                                args_ff=False, args_cof=False, args_eof=True)

        return torch.cat([f_cls_vector, f_arg1_vec, f_arg2_vec, f_arg1_vec * f_arg2_vec], dim=1), torch.cat([co_cls_vector], dim=1), torch.cat([eo_arg1_vec, eo_arg2_vec, eo_arg1_vec * eo_arg2_vec], dim=1)

    def frozen_forward(self, input_):
        return self.linear(input_)

    def forward(self, f_input_ids, f_attention_mask, f_position_ids, f_global_attention_mask, f_arg1, f_arg2,
                    co_input_ids, co_attention_mask, co_position_ids, co_global_attention_mask,
                    eo_input_ids, eo_attention_mask, eo_position_ids,
                    lm_only=False, pre_lm_out=False):

        # if pre_lm_out:
        #     return self.linear(f_input_ids), self.linear(co_input_ids), self.linear(eo_input_ids)

        f_lm_output, c_lm_output, e_lm_output = self.generate_model_output(f_input_ids, f_attention_mask, f_position_ids, f_global_attention_mask, f_arg1, f_arg2,
                                                co_input_ids, co_attention_mask, co_position_ids, co_global_attention_mask,
                                                eo_input_ids, eo_attention_mask, eo_position_ids)



        if lm_only:
            return f_lm_output, c_lm_output, e_lm_output

        return self.linear(f_lm_output), self.c_linear(c_lm_output), self.e_linear(e_lm_output)