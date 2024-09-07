from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import string
import os
import numpy as np
import torch
import re

TRAIN = 'train'
DEV = 'dev'
TEST = 'test'


def load_lemma_dataset(tsv_path, force_balance=False):
    all_examples = []
    label_map = {'POS': 1, 'NEG': 0}
    with open(tsv_path) as tnf:
        for line in tnf:
            row = line.strip().split('\t')
            mention_pair = row[:2]
            label = label_map[row[2]]
            all_examples.append((mention_pair, label))
    if force_balance:
        from collections import defaultdict
        import random
        random.seed(42)
        label2eg = defaultdict(list)

        for eg in all_examples:
            label2eg[eg[1]].append(eg)

        min_label = min(label2eg.keys(), key=lambda x: len(label2eg[x]))
        min_label_len = len(label2eg[min_label])

        max_eg_len = max([len(val) for val in label2eg.values()])
        random_egs = random.choices(label2eg[min_label], k=max_eg_len - min_label_len)
        all_examples.extend(random_egs)

        label2eg = defaultdict(list)

        for eg in all_examples:
            label2eg[eg[1]].append(eg)

        # print([len(val) for val in label2eg.values()])

    return all_examples


def get_arg_attention_mask(input_ids, parallel_model):
    """
    Get the global attention mask and the indices corresponding to the tokens between
    the mention indicators.
    Parameters
    ----------
    input_ids
    parallel_model

    Returns
    -------
    Tensor, Tensor, Tensor
        The global attention mask, arg1 indicator, and arg2 indicator
    """
    input_ids.cpu()

    num_inputs = input_ids.shape[0]  # len(batch_size)

    m_start_indicator = input_ids == parallel_model.module.start_id
    m_end_indicator = input_ids == parallel_model.module.end_id

    m = m_start_indicator + m_end_indicator  # 得到事件触发词所在位置，其他地方为false

    # non-zero indices are the tokens corresponding to <m> and </m>
    nz_indexes = m.nonzero()[:, 1].reshape((num_inputs, 4))  # 得到<m>和</m>所在tokens序列中的索引

    # Now we need to make the tokens between <m> and </m> to be non-zero
    q = torch.arange(m.shape[1])
    q = q.repeat(m.shape[0], 1)  # (batch_size, 512)

    # all indices greater than and equal to the first <m> become True
    msk_0 = (nz_indexes[:, 0].repeat(m.shape[1], 1).transpose(0, 1)) <= q
    # all indices less than and equal to the first </m> become True
    msk_1 = (nz_indexes[:, 1].repeat(m.shape[1], 1).transpose(0, 1)) >= q
    # all indices greater than and equal to the second <m> become True
    msk_2 = (nz_indexes[:, 2].repeat(m.shape[1], 1).transpose(0, 1)) <= q
    # all indices less than and equal to the second </m> become True
    msk_3 = (nz_indexes[:, 3].repeat(m.shape[1], 1).transpose(0, 1)) >= q

    # excluding <m> and </m> gives only the indices between <m> and </m>
    msk_0_ar = (nz_indexes[:, 0].repeat(m.shape[1], 1).transpose(0, 1)) < q
    msk_1_ar = (nz_indexes[:, 1].repeat(m.shape[1], 1).transpose(0, 1)) > q
    msk_2_ar = (nz_indexes[:, 2].repeat(m.shape[1], 1).transpose(0, 1)) < q
    msk_3_ar = (nz_indexes[:, 3].repeat(m.shape[1], 1).transpose(0, 1)) > q

    # Union of indices between first <m> and </m> and second <m> and </m>
    attention_mask_g = msk_0.int() * msk_1.int() + msk_2.int() * msk_3.int()
    # attention_mask_g = None
    # attention_mask_g[:, 0] = 1

    # indices between <m> and </m> excluding the <m> and </m>
    arg1 = msk_0_ar.int() * msk_1_ar.int()
    arg2 = msk_2_ar.int() * msk_3_ar.int()

    return attention_mask_g, arg1, arg2


def forward_ab(parallel_model, ab_dict, device, indices, lm_only=False):
    batch_tensor_ab = ab_dict['input_ids'][indices, :]
    batch_am_ab = ab_dict['attention_mask'][indices, :]
    batch_posits_ab = ab_dict['position_ids'][indices, :]
    am_g_ab, arg1_ab, arg2_ab = get_arg_attention_mask(batch_tensor_ab, parallel_model)

    batch_tensor_ab.to(device)
    batch_am_ab.to(device)
    batch_posits_ab.to(device)
    if am_g_ab is not None:
        am_g_ab.to(device)
    arg1_ab.to(device)
    arg2_ab.to(device)

    return parallel_model(batch_tensor_ab, attention_mask=batch_am_ab, position_ids=batch_posits_ab,
                          global_attention_mask=am_g_ab, arg1=arg1_ab, arg2=arg2_ab, lm_only=lm_only)

def c_only_forward_ab(c_only_parallel_model, ab_dict, device, indices, lm_only=False):
    batch_tensor_ab = ab_dict['input_ids'][indices, :]
    batch_am_ab = ab_dict['attention_mask'][indices, :]
    batch_posits_ab = ab_dict['position_ids'][indices, :]
    # am_g_ab, arg1_ab, arg2_ab = get_arg_attention_mask(batch_tensor_ab, c_only_parallel_model)

    # 获取 <mask> token 的 ID
    mask_token_id = c_only_parallel_model.module.tokenizer.mask_token_id
    # 生成 mask_attention: 如果 token 不是 <mask>，则为 1，否则为 0
    mask_attention = (batch_tensor_ab != mask_token_id).long()

    batch_tensor_ab.to(device)
    batch_am_ab.to(device)
    batch_posits_ab.to(device)
    if mask_attention is not None:
        mask_attention.to(device)

    return c_only_parallel_model(batch_tensor_ab, attention_mask=batch_am_ab, position_ids=batch_posits_ab,
                          global_attention_mask=mask_attention, lm_only=lm_only)

def e_only_forward_ab(e_only_parallel_model, ab_dict, device, indices, lm_only=False):
    batch_tensor_ab = ab_dict['input_ids'][indices, :]
    batch_am_ab = ab_dict['attention_mask'][indices, :]
    batch_posits_ab = ab_dict['position_ids'][indices, :]
    # am_g_ab, arg1_ab, arg2_ab = get_arg_attention_mask(batch_tensor_ab, e_only_parallel_model)

    batch_tensor_ab.to(device)
    batch_am_ab.to(device)
    batch_posits_ab.to(device)

    return e_only_parallel_model(batch_tensor_ab, attention_mask=batch_am_ab, position_ids=batch_posits_ab, lm_only=lm_only)

def tokenize(tokenizer, mention_pairs, mention_map, m_end, max_sentence_len=1024, text_key='bert_doc', truncate=True):
    if max_sentence_len is None:
        max_sentence_len = tokenizer.model_max_length  # 512

    pairwise_bert_instances_ab = []
    pairwise_bert_instances_ba = []

    c_only_pairwise_bert_instances_ab = []
    c_only_pairwise_bert_instances_ba = []

    e_only_pairwise_ab = []
    e_only_pairwise_ba = []

    doc_start = '<doc-s>'
    doc_end = '</doc-s>'

    for (m1, m2) in mention_pairs:
        sentence_a = mention_map[m1][text_key]  # text_key=bert_sentence
        sentence_b = mention_map[m2][text_key]
        e_only_a = mention_map[m1]['mention_text']
        e_only_b = mention_map[m2]['mention_text']

        e_only_pairwise_ab.append((e_only_a, e_only_b))
        e_only_pairwise_ba.append((e_only_b, e_only_a))

        def make_instance(sent_a, sent_b):
            return ' '.join(['<g>', doc_start, sent_a, doc_end]), \
                ' '.join([doc_start, sent_b, doc_end])

        def make_sentence_c_only(sent_a, sent_b):
            masked_sentence_a = re.sub(r'<m>.*?</m>', '<mask>', sent_a)
            masked_sentence_b = re.sub(r'<m>.*?</m>', '<mask>', sent_b)
            mask_instance_ab = make_instance(masked_sentence_a, masked_sentence_b)
            mask_instance_ba = make_instance(masked_sentence_b, masked_sentence_a)
            return mask_instance_ab, mask_instance_ba

        instance_ab = make_instance(sentence_a, sentence_b)  # 得到所有提及对的句子
        pairwise_bert_instances_ab.append(instance_ab)

        instance_ba = make_instance(sentence_b, sentence_a)
        pairwise_bert_instances_ba.append(instance_ba)

        c_only_instance_ab, c_only_instance_ba = make_sentence_c_only(sentence_a, sentence_b)
        c_only_pairwise_bert_instances_ab.append(c_only_instance_ab)
        c_only_pairwise_bert_instances_ba.append(c_only_instance_ba)

    def truncate_with_mentions(input_ids):
        input_ids_truncated = []
        for input_id in input_ids:
            m_end_index = input_id.index(m_end)  # model.end_id在当前input_ids中所在位置的索引

            curr_start_index = max(0, m_end_index - (max_sentence_len // 4))

            in_truncated = input_id[curr_start_index: m_end_index] + \
                           input_id[
                           m_end_index: m_end_index + (max_sentence_len // 4)]  # 截取编码当中的一部分，end_ids向前向后各看512//4步
            in_truncated = in_truncated + [tokenizer.pad_token_id] * (max_sentence_len // 2 - len(in_truncated))  # 256
            input_ids_truncated.append(in_truncated)

        return torch.LongTensor(input_ids_truncated)

    def ab_tokenized(pair_wise_instances, c_only_pairwise_bert_instances, e_only_pairwise):
        instances_a, instances_b = zip(*pair_wise_instances)  # pair_wise_instances列表中保存的都是提及句子对的元组
        mask_instance_a, mask_instance_b = zip(*c_only_pairwise_bert_instances)
        e_only_a, e_only_b = zip(*e_only_pairwise)

        tokenized_a = tokenizer(list(instances_a), add_special_tokens=False)  # 对instance_a中的所有句子进行编码，得到input_ids和attention_mask（句子部分为1）
        tokenized_b = tokenizer(list(instances_b), add_special_tokens=False)

        c_only_tokenized_a = tokenizer(list(mask_instance_a), add_special_tokens=False, max_length=256, truncation=True, padding='max_length')['input_ids']
        c_only_tokenized_a = torch.LongTensor(c_only_tokenized_a)
        c_only_positions_a = torch.arange(c_only_tokenized_a.shape[-1]).expand(c_only_tokenized_a.shape)
        c_only_tokenized_b = tokenizer(list(mask_instance_b), add_special_tokens=False, max_length=256, truncation=True, padding='max_length')['input_ids']
        c_only_tokenized_b = torch.LongTensor(c_only_tokenized_b)
        c_only_positions_b = torch.arange(c_only_tokenized_b.shape[-1]).expand(c_only_tokenized_b.shape)

        e_only_tokenized_a = tokenizer(list(e_only_a), add_special_tokens=False, max_length=256, truncation=True, padding='max_length')['input_ids']
        e_only_tokenized_a = torch.LongTensor(e_only_tokenized_a)
        e_only_positions_a = torch.arange(e_only_tokenized_a.shape[-1]).expand(e_only_tokenized_a.shape)
        e_only_tokenized_b = tokenizer(list(e_only_b), add_special_tokens=False, max_length=256, truncation=True, padding='max_length')['input_ids']
        e_only_tokenized_b = torch.LongTensor(e_only_tokenized_b)
        e_only_positions_b = torch.arange(e_only_tokenized_b.shape[-1]).expand(e_only_tokenized_b.shape)

        tokenized_a = truncate_with_mentions(tokenized_a['input_ids'])  # (27928, 256)对tokenizer_a的input_ids中的每个input_id进行截断处理，从end_id所在位置向前向后看512/4步，然后pad到256长度
        positions_a = torch.arange(tokenized_a.shape[-1]).expand(tokenized_a.shape)
        tokenized_b = truncate_with_mentions(tokenized_b['input_ids'])
        positions_b = torch.arange(tokenized_b.shape[-1]).expand(tokenized_b.shape)  # (27928, 256)

        tokenized_ab_ = torch.hstack((tokenized_a, tokenized_b))  # (27928, 256+256)-->(27928, 512)
        positions_ab = torch.hstack((positions_a, positions_b))

        c_only_tokenized_ab = torch.hstack((c_only_tokenized_a, c_only_tokenized_b))
        c_only_positions_ab = torch.hstack((c_only_positions_a, c_only_positions_b))

        e_only_tokenized_ab = torch.hstack((e_only_tokenized_a, e_only_tokenized_b))
        e_only_positions_ab = torch.hstack((e_only_positions_a, e_only_positions_b))

        tokenized_ab_dict = {'input_ids': tokenized_ab_,
                             'attention_mask': (tokenized_ab_ != tokenizer.pad_token_id),
                             'position_ids': positions_ab
                             }

        c_only_tokenized_ab_dict = {'input_ids': c_only_tokenized_ab,
                                    'attention_mask': (c_only_tokenized_ab != tokenizer.pad_token_id),
                                    'position_ids': c_only_positions_ab
                                    }

        e_only_tokenized_ab_dict = {'input_ids': e_only_tokenized_ab,
                                    'attention_mask': (e_only_tokenized_ab != tokenizer.pad_token_id),
                                    'position_ids': e_only_positions_ab
                                    }

        return tokenized_ab_dict, c_only_tokenized_ab_dict, e_only_tokenized_ab_dict

    if truncate:
        tokenized_ab, c_only_tokenized_ab, e_only_tokenized_ab = ab_tokenized(pairwise_bert_instances_ab, c_only_pairwise_bert_instances_ab, e_only_pairwise_ab)  # 得到input_ids、attention_mask、position_ids，在处理过程中，是对两个句子分别处理最后堆叠到一起
        tokenized_ba, c_only_tokenized_ba, e_only_tokenized_ba = ab_tokenized(pairwise_bert_instances_ba, c_only_pairwise_bert_instances_ba, e_only_pairwise_ba)
    else:
        instances_ab = [' '.join(instance) for instance in pairwise_bert_instances_ab]
        instances_ba = [' '.join(instance) for instance in pairwise_bert_instances_ba]
        tokenized_ab = tokenizer(list(instances_ab), add_special_tokens=False, padding=True)

        tokenized_ab_input_ids = torch.LongTensor(tokenized_ab['input_ids'])

        tokenized_ab = {'input_ids': torch.LongTensor(tokenized_ab['input_ids']),
                        'attention_mask': torch.LongTensor(tokenized_ab['attention_mask']),
                        'position_ids': torch.arange(tokenized_ab_input_ids.shape[-1]).expand(
                            tokenized_ab_input_ids.shape)}

        tokenized_ba = tokenizer(list(instances_ba), add_special_tokens=False, padding=True)
        tokenized_ba_input_ids = torch.LongTensor(tokenized_ba['input_ids'])
        tokenized_ba = {'input_ids': torch.LongTensor(tokenized_ba['input_ids']),
                        'attention_mask': torch.LongTensor(tokenized_ba['attention_mask']),
                        'position_ids': torch.arange(tokenized_ba_input_ids.shape[-1]).expand(
                            tokenized_ba_input_ids.shape)}

    return tokenized_ab, tokenized_ba, c_only_tokenized_ab, c_only_tokenized_ba, e_only_tokenized_ab, e_only_tokenized_ba


def cluster_cc(affinity_matrix, threshold=0.8):
    """
    Find connected components using the affinity matrix and threshold -> adjacency matrix
    Parameters
    ----------
    affinity_matrix: np.array
    threshold: float

    Returns
    -------
    list, np.array
    """
    adjacency_matrix = csr_matrix(affinity_matrix > threshold)
    clusters, labels = connected_components(adjacency_matrix, return_labels=True, directed=False)
    return clusters, labels


def remove_puncts(target_str):
    return target_str
    # return target_str.translate(str.maketrans('', '', string.punctuation)).lower()


def jc(arr1, arr2):
    return len(set.intersection(arr1, arr2)) / len(set.union(arr1, arr2))


def generate_mention_pairs(mention_map, split):
    """

    Parameters
    ----------
    mention_map: dict
    split: str (train/dev/test)

    Returns
    -------
    list: A list of all possible mention pairs within a topic
    """
    split_mention_ids = sorted([m_id for m_id, m in mention_map.items() if
                                m['split'] == split])  # 按train、dev、test分别从当前数据集中读取对应的数据，这里是事件类型为evt的数据
    topic2mentions = {}
    for m_id in split_mention_ids:  # 得到某一主题下的所有mention_id，每个topic对应一个列表，存储该topic下的mention_id
        try:
            topic = mention_map[m_id]['predicted_topic']  # specifically for the test set of ECB
        except KeyError:
            topic = None
        if not topic:
            topic = mention_map[m_id]['topic']
        if topic not in topic2mentions:
            topic2mentions[topic] = []
        topic2mentions[topic].append(m_id)
    # 训练集中有25个topic，每个topic下都分别存储了事件提及id
    mention_pairs = []  # 在每个主题中构造事件提及对，列表中存储元组[( )]

    for mentions in topic2mentions.values():
        list_mentions = list(mentions)
        for i in range(len(list_mentions)):
            for j in range(i + 1):
                if i != j:
                    mention_pairs.append((list_mentions[i], list_mentions[j]))

    return mention_pairs  # 在每一个主题中构造提及对，分别构造训练集、验证集和测试集的提及对


def generate_key_file(coref_map_tuples, name, out_dir, out_file_path):
    """

    Parameters
    ----------
    coref_map_tuples: list
    name: str
    out_dir: str
    out_file_path: str

    Returns
    -------
    None
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    clus_to_int = {}
    clus_number = 0
    with open(out_file_path, 'w') as of:
        of.write("#begin document (%s);\n" % name)
        for i, map_ in enumerate(coref_map_tuples):
            en_id = map_[0]
            clus_id = map_[1]
            if clus_id in clus_to_int:
                clus_int = clus_to_int[clus_id]
            else:
                clus_to_int[clus_id] = clus_number
                clus_number += 1
                clus_int = clus_to_int[clus_id]  # 聚类的索引
            of.write("%s\t0\t%d\t%s\t(%d)\n" % (name, i, en_id, clus_int))  # 当前事件提及所在的聚类的索引
        of.write("#end document\n")


def cluster(mentions, mention_pairs, similarities, threshold=0):
    n = len(mentions)
    m_id2ind = {m: i for i, m in enumerate(mentions)}

    mention_ind_pairs = [(m_id2ind[mp[0]], m_id2ind[mp[1]]) for mp in mention_pairs]
    rows, cols = zip(*mention_ind_pairs)

    # create similarity matrix from the similarities
    n = len(mentions)
    similarity_matrix = np.identity(n)
    similarity_matrix[rows, cols] = similarities

    clusters, labels = cluster_cc(similarity_matrix, threshold=threshold)
    m_id2cluster = {m: i for m, i in zip(mentions, labels)}
    return m_id2cluster


def accuracy(predicted_labels, true_labels):
    """
    Accuracy is correct predictions / all predicitons
    """
    return sum(predicted_labels == true_labels) / len(predicted_labels)


def precision(predicted_labels, true_labels):
    """
    Precision is True Positives / All Positives Predictions
    """
    return sum(torch.logical_and(predicted_labels, true_labels)) / sum(predicted_labels)


def recall(predicted_labels, true_labels):
    """
    Recall is True Positives / All Positive Labels
    """
    return sum(torch.logical_and(predicted_labels, true_labels)) / sum(true_labels)


def f1_score(predicted_labels, true_labels):
    """
    F1 score is the harmonic mean of precision and recall
    """
    P = precision(predicted_labels, true_labels)
    R = recall(predicted_labels, true_labels)
    return 2 * P * R / (P + R)


def save_parameters(scorer_folder, parallel_model, c_only_parallel_model, e_only_parallel_model):
    model_path = scorer_folder + 'f_CrossEncoder/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path_linear = model_path + 'linear.chkpt'
    torch.save(parallel_model.module.linear.state_dict(), model_path_linear)
    parallel_model.module.model.save_pretrained(model_path + '/bert')
    parallel_model.module.tokenizer.save_pretrained(model_path + '/bert')

    model_path_c_only = scorer_folder + 'c_only_CrossEncoder/'
    if not os.path.exists(model_path_c_only):
        os.makedirs(model_path_c_only)
    model_path_c_only_linear = model_path_c_only + 'linear.chkpt'
    torch.save(c_only_parallel_model.module.linear.state_dict(), model_path_c_only_linear)
    c_only_parallel_model.module.model.save_pretrained(model_path_c_only + '/bert')
    c_only_parallel_model.module.tokenizer.save_pretrained(model_path_c_only + '/bert')


    model_path_e_only = scorer_folder + 'e_only_CrossEncoder/'
    if not os.path.exists(model_path_e_only):
        os.makedirs(model_path_e_only)
    model_path_e_only_linear = model_path_e_only + 'linear.chkpt'
    torch.save(e_only_parallel_model.module.linear.state_dict(), model_path_e_only_linear)
    e_only_parallel_model.module.model.save_pretrained(model_path_e_only + '/bert')
    e_only_parallel_model.module.tokenizer.save_pretrained(model_path_e_only + '/bert')
