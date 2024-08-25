import pickle
from helper import *
import string
from tqdm import tqdm
import numpy as np


def get_mention_pair_similarity_lemma2(mention_pairs, mention_map, relations, threshold = 0.05):
    """
    Generate the similarities for mention pairs

    Parameters
    ----------
    mention_pairs: list
    mention_map: dict
    relations: list
        The list of relations represented as a triple: (head, label, tail)
    working_folder: str

    Returns
    -------
    list
    """
    similarities = []

    within_doc_similarities = []

    # doc_sent_map = pickle.load(open(working_folder + '/doc_sent_map.pkl', 'rb'))
    # doc_sims = pickle.load(open(working_folder + '/doc_sims_path.pkl', 'rb'))
    doc_ids = []

    # for doc_id, _ in list(doc_sent_map.items()):
    #     doc_ids.append(doc_id)

    doc2id = {doc: i for i, doc in enumerate(doc_ids)}

    # generate similarity using the mention text
    for pair in tqdm(mention_pairs, desc='Generating Similarities'):
        men1, men2 = pair
        men_map1 = mention_map[men1]
        men_map2 = mention_map[men2]
        men_text1 = men_map1['mention_text'].lower()
        men_text2 = men_map2['mention_text'].lower()

        def jc(arr1, arr2):  # 用于计算两个集合之间相似度的度量，通常称为 Jaccard 相似系数（Jaccard similarity coefficient）
            return len(set.intersection(arr1, arr2))/len(set.union(arr1, arr2))  # 两个文本之间交集的长度除以并集的长度，反映两个文本之间的相似度
            # return len(set.intersection(arr1, arr2))

        doc_id1 = men_map1['doc_id']
        # sent_id1 = int(men_map1['sentence_id'])
        # all_sent_ids1 = {str(sent_id1 - 1), str(sent_id1), str(sent_id1 + 1)}
        # all_sent_ids1 = {str(sent_id1)}
        #
        # doc_id2 = men_map2['doc_id']
        # sent_id2 = int(men_map2['sentence_id'])
        # all_sent_ids2 = {str(sent_id2 - 1), str(sent_id2), str(sent_id2 + 1)}
        #
        # all_sent_ids2 = {str(sent_id2)}

        # sentence_tokens1 = [tok for sent_id in all_sent_ids1 if sent_id in doc_sent_map[doc_id1]
        #                     for tok in doc_sent_map[doc_id1][sent_id]['sentence_tokens']]
        #
        # sentence_tokens2 = [tok for sent_id in all_sent_ids2 if sent_id in doc_sent_map[doc_id2]
        #                     for tok in doc_sent_map[doc_id2][sent_id]['sentence_tokens']]

        sentence_tokens1 = [tok for tok in men_map1['sentence_tokens']]

        sentence_tokens2 = [tok for tok in men_map2['sentence_tokens']]

        sent_sim = jc(set(sentence_tokens1), set(sentence_tokens2))  # 比较两个句子之间的相似度
        # sent_sim = jc(set(men_map1['sentence_tokens']), set(men_map2['sentence_tokens']))
        # doc_sim = doc_sims[doc2id[men_map1['doc_id']], doc2id[men_map2['doc_id']]]
        lemma_sim = float(men_map1['lemma'].lower() in men_text2 or men_map2['lemma'].lower() in men_text1
                          or men_map1['lemma'].lower() in men_map2['lemma'].lower()
                          )

        lemma1 = men_map1['lemma'].lower()
        lemma2 = men_map2['lemma'].lower()
        if lemma1 > lemma2:
            pair_tuple = (lemma2, lemma1)
        else:
            pair_tuple = (lemma1, lemma2)

        # similarities.append((lemma_sim or pair_tuple in relations))
        similarities.append((lemma_sim or pair_tuple in relations) and sent_sim > threshold)  # 根据提及所在的句子之间的相似性以及词元之间的相似性来判断当前提及对的相似度，返回值为False和True
        # similarities.append((lemma_sim) and sent_sim > 0.05)
        # similarities.append((lemma_sim + 0.3*sent_sim)/2)




    return np.array(similarities)


def get_mention_pair_similarity_lemma(mention_pairs, mention_map, syn_lemma_pairs, threshold=0.05, doc_sent_map=None):
    similarities = []

    # generate similarity using the mention text
    for pair in tqdm(mention_pairs, desc='Generating Similarities'):
        men1, men2 = pair
        men_map1 = mention_map[men1]
        men_map2 = mention_map[men2]
        men_text1 = remove_puncts(men_map1['mention_text'].lower())
        men_text2 = remove_puncts(men_map2['mention_text'].lower())
        lemma1 = remove_puncts(men_map1['lemma'].lower())
        lemma2 = remove_puncts(men_map2['lemma'].lower())

        # doc_id1 = men_map1['doc_id']
        # sent_id1 = int(men_map1['sentence_id'])
        # all_sent_ids1 = {str(sent_id1 - 1), str(sent_id1), str(sent_id1 + 1)}
        # all_sent_ids1 = {str(sent_id1)}
        #
        # doc_id2 = men_map2['doc_id']
        # sent_id2 = int(men_map2['sentence_id'])
        # all_sent_ids2 = {str(sent_id2 - 1), str(sent_id2), str(sent_id2 + 1)}
        #
        # all_sent_ids2 = {str(sent_id2)}

        # sentence_tokens1 = [tok for sent_id in all_sent_ids1 if sent_id in doc_sent_map[doc_id1]
        #                     for tok in doc_sent_map[doc_id1][sent_id]['sentence_tokens']]
        #
        # sentence_tokens2 = [tok for sent_id in all_sent_ids2 if sent_id in doc_sent_map[doc_id2]
        #                     for tok in doc_sent_map[doc_id2][sent_id]['sentence_tokens']]

        sentence_tokens1 = [tok.lower() for tok in men_map1['sentence_tokens']]

        sentence_tokens2 = [tok.lower() for tok in men_map2['sentence_tokens']]

        sent_sim = jc(set(sentence_tokens1), set(sentence_tokens2))
        # sent_sim = jc(set(men_map1['sentence_tokens']), set(men_map2['sentence_tokens']))
        # doc_sim = doc_sims[doc2id[men_map1['doc_id']], doc2id[men_map2['doc_id']]]
        lemma_sim = float(lemma1 in men_text2 or lemma2 in men_text1
                          or men_text1 in lemma2
                          )
        pair_tuple = tuple(sorted([lemma1, lemma2]))

        similarities.append((lemma_sim or pair_tuple in syn_lemma_pairs) and sent_sim > threshold)

    return np.array(similarities)


def get_all_mention_pairs_labels_split(mention_map, split):
    split_mention_pairs = generate_mention_pairs(mention_map, split)  # 在每一个主题中构造提及对，分别构造训练集、验证集和测试集的提及对
    split_labels = [int(mention_map[m1]['gold_cluster'] == mention_map[m2]['gold_cluster']) for m1, m2 in
                    split_mention_pairs]  # 得到提及对的标签，gold_cluster相同的为1，不同为0， 训练集中总共有341706个事件对
    split_pairs_labels = list(zip(split_mention_pairs, split_labels))  # 将事件对与标签label对应起来
    return split_pairs_labels


def get_all_mention_pairs_labels(mention_map):
    all_mention_pairs_labels = []
    for split in [TRAIN, DEV, TEST]:
        split_pairs_labels = get_all_mention_pairs_labels_split(mention_map, split)  # 分别得到训练集、验证集和测试集的在同一topic中的事件对以及对应的标签
        all_mention_pairs_labels.append(split_pairs_labels) # 训练集有341706个事件提及对，验证集有100784个事件提及对，测试集有93878个事件提及对
    return all_mention_pairs_labels


def get_lemma_pairs_labels(mention_map, pairs_labels):
    lemma_pairs_labels = []
    for (m1, m2), label in pairs_labels:
        lemma1 = remove_puncts(mention_map[m1]['lemma'].lower())  # 简单理解一下，相当于事件触发词吧
        lemma2 = remove_puncts(mention_map[m2]['lemma'].lower())
        if lemma1 > lemma2:  # 通过字典序（即字母顺序）来进行比较
            pair_tuple = (lemma2, lemma1)
        else:
            pair_tuple = (lemma1, lemma2)

        # lemma_pair = tuple(sorted([remove_puncts(mention_map[m1]['lemma'].lower()),
        #                            remove_puncts(mention_map[m2]['lemma'].lower())]))
        lemma_pairs_labels.append((pair_tuple, label))
    return lemma_pairs_labels


def generate_tp_fp_tn_fn(mention_pairs, ground_truth, mention_map, syn_lemma_pairs, threshold=0.05, doc_sent_map=None):
    similarities = get_mention_pair_similarity_lemma2(mention_pairs, mention_map, syn_lemma_pairs,
                                                     threshold=threshold)  # 根据提及所在的句子之间的相似性以及词元之间的相似性来判断当前提及对的相似度，返回值为False和True

    lemma_coref = similarities > 0.15
    # print('all positives:', lemma_coref.sum())

    tps = np.logical_and(lemma_coref, ground_truth).nonzero()   # 得到相似性与真实标签之间预测为1的位置索引
    tps = [mention_pairs[i] for i in tps[0]]  # 13065个
    fps = np.logical_and(lemma_coref, np.logical_not(ground_truth)).nonzero()  # 提取出把不共指的预测成共指的提及对，应该是负样本中的hard
    fps = [mention_pairs[i] for i in fps[0]]  # 14863个
    tns = np.logical_and(np.logical_not(lemma_coref), np.logical_not(ground_truth)).nonzero()  # 提取预测的是负样本的提及对，标签也是负样本
    tns = [mention_pairs[i] for i in tns[0]]  # 311632个
    fns = np.logical_and(np.logical_not(lemma_coref), ground_truth).nonzero()  # 把共指的预测成不共指的 P+_FN
    fns = [mention_pairs[i] for i in fns[0]]  # 2146个

    print('true positives:', len(tps))
    print('false positives:', len(fps))
    print('true negatives:', len(tns))
    print('false negatives:', len(fns))

    ind2m_id = list(mention_map.keys())
    n = len(ind2m_id)
    m_id2ind = {m: i for i, m in enumerate(ind2m_id)}
    sim_matrix = np.zeros((n, n))
    for ((m1, m2), sim) in zip(mention_pairs, similarities):
        sim_matrix[m_id2ind[m1], m_id2ind[m2]] = sim
    clusters, labels = cluster_cc(sim_matrix, threshold=0.15)
    m_id2cluster = {m: i for m, i in zip(ind2m_id, labels)}
    lemma_coref_transitive = np.array([m_id2cluster[m1] == m_id2cluster[m2] for m1, m2 in mention_pairs])

    tps_trans = np.logical_and(lemma_coref_transitive, ground_truth).nonzero()
    tps_trans = [mention_pairs[i] for i in tps_trans[0]]  # 15189
    fps_trans = np.logical_and(lemma_coref_transitive, np.logical_not(ground_truth)).nonzero()
    fps_trans = [mention_pairs[i] for i in fps_trans[0]]  # 65358
    tns_trans = np.logical_and(np.logical_not(lemma_coref_transitive), np.logical_not(ground_truth)).nonzero()
    tns_trans = [mention_pairs[i] for i in tns_trans[0]]  # 261137
    fns_trans = np.logical_and(np.logical_not(lemma_coref_transitive), ground_truth).nonzero()
    fns_trans = [mention_pairs[i] for i in fns_trans[0]]  # 22

    print('\nAfter transitive closure\ntrue positives:', len(tps_trans))
    print('false positives:', len(fps_trans))
    print('true negatives:', len(tns_trans))
    print('false negatives:', len(fns_trans))
    return (tps, fps, tns, fns), (tps_trans, fps_trans, tns_trans, fns_trans)


def lh(dataset, threshold=0.05):
    """

    Parameters
    ----------
    dataset: str
        The dataset name: ecb/gvc
    threshold: double

    Returns
    -------
    None: Save the predicted mention pairs from the dataset in the dataset's folder
        Directory location: ./datasets/dataset/lh/
    """
    dataset_folder = f'./datasets/{dataset}/'
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))  # 读取数据集
    evt_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt'}  # 将mention_map中的所有men_type为evt的数据提取出来
    tr_mention_pairs_labels, dev_mention_pairs_labels, test_mention_pairs_labels = get_all_mention_pairs_labels(evt_mention_map)  # 得到事件提及对和对应的标签

    train_lemma_pairs_labels = get_lemma_pairs_labels(evt_mention_map, tr_mention_pairs_labels)  # 相当于返回训练集事件提及对中的事件触发词对与对应标签

    train_syn_lemma_pairs = set([p for p, l in train_lemma_pairs_labels if l == 1])  # 根据标签为1，返回词元对，词元对中的词相当于同义词
    train_non_syn_pairs = set([p for p, l in train_lemma_pairs_labels if l == 0 and p not in train_syn_lemma_pairs])  # 返回不是同义词的对

    # train_syn_lemma_pls = [(p, l) for p, l in train_lemma_pairs_labels if p in train_syn_lemma_pairs]
    # train_non_syn_lps = [(p, l) for p, l in train_lemma_pairs_labels if p in train_non_syn_pairs]

    for split, pair_labels in zip([TRAIN, DEV, TEST], [tr_mention_pairs_labels, dev_mention_pairs_labels, test_mention_pairs_labels]):
        print(split)  # train--> ((事件提及对)和标签)  dev--> ((事件提及对)和标签)  test--> ((事件提及对)和标签)
        pairs, labels = zip(*pair_labels)  # ((事件提及对)和标签)-->(事件提及对) (labels)
        (mps, mps_trans) = generate_tp_fp_tn_fn(pairs, np.array(labels), mention_map, train_syn_lemma_pairs, threshold=threshold)
        pickle.dump((mps, mps_trans), open(f'./datasets/{dataset}/lh/mp_mp_t_{split}.pkl', 'wb'))


def lh_oracle(dataset, threshold=0.05):
    """

    Parameters
    ----------
    dataset: str
        The dataset name: ecb/gvc
    threshold: double

    Returns
    -------
    None: Save the predicted mention pairs from the dataset in the dataset's folder
        Directory location: ./datasets/dataset/lh_oracle/
    """
    dataset_folder = f'./datasets/{dataset}/'
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    evt_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt'}
    tr_mention_pairs_labels, dev_mention_pairs_labels, test_mention_pairs_labels = get_all_mention_pairs_labels(
        evt_mention_map)

    train_syn_lemma_pairs = get_lemma_pairs_labels(evt_mention_map, tr_mention_pairs_labels)
    dev_syn_lemma_pairs =   get_lemma_pairs_labels(evt_mention_map, dev_mention_pairs_labels)
    test_syn_lemma_pairs =  get_lemma_pairs_labels(evt_mention_map, test_mention_pairs_labels)

    tr_syn_lemma_pairs = set([p for p, l in train_syn_lemma_pairs if l == 1])
    dev_syn_lemma_pairs = set([p for p, l in dev_syn_lemma_pairs if l == 1])
    test_syn_lemma_pairs = set([p for p, l in test_syn_lemma_pairs if l == 1])

    split_syn_lemma = {split: syns for split, syns in zip([TRAIN, DEV, TEST], [tr_syn_lemma_pairs, dev_syn_lemma_pairs, test_syn_lemma_pairs])}

    all_syn_lemmas = tr_syn_lemma_pairs.union(dev_syn_lemma_pairs).union(test_syn_lemma_pairs)

    pass
    for split, pair_labels in zip([TRAIN, DEV, TEST], [tr_mention_pairs_labels, dev_mention_pairs_labels, test_mention_pairs_labels]):
        print('-------', split, '--------')
        pairs, labels = zip(*pair_labels)
        (mps, mps_trans) = generate_tp_fp_tn_fn(pairs, np.array(labels), mention_map, split_syn_lemma[split], threshold=threshold)
        pickle.dump((mps, mps_trans), open(f'./datasets/{dataset}/lh_oracle/mp_mp_t_{split}.pkl', 'wb'))


def lh_split(heu, dataset, split, threshold=0.05):
    dataset_folder = f'./datasets/{dataset}/'
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    evt_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt'}
    split_mention_pairs_labels = get_all_mention_pairs_labels_split(evt_mention_map, split)

    if heu == 'lh':
        train_menrion_pairs_labels = get_all_mention_pairs_labels_split(evt_mention_map, 'train')
        train_syn_lemma_pairs = get_lemma_pairs_labels(evt_mention_map, train_menrion_pairs_labels)
        split_syn_lemma_pairs = set([p for p, l in train_syn_lemma_pairs if l == 1])
    else:
        split_syn_lemma_pairs = get_lemma_pairs_labels(evt_mention_map, split_mention_pairs_labels)
        split_syn_lemma_pairs = set([p for p, l in split_syn_lemma_pairs if l == 1])

    pairs, labels = zip(*split_mention_pairs_labels)
    (mps, mps_trans) = generate_tp_fp_tn_fn(pairs, np.array(labels), mention_map, split_syn_lemma_pairs,
                                            threshold=threshold)
    return mps, mps_trans


if __name__ == '__main__':
    lh('ecb', threshold=0.05)
    # lh_oracle('gvc', threshold=0)
    # print('------- lh -------')
    # lh('gvc', threshold=0.04)
    # print('------- lh oracle -------')
    # lh_oracle('gvc', threshold=0.04)
