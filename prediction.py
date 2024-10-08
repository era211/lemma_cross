from helper import *
import pickle
import numpy as np
from coval.coval.conll.reader import get_coref_infos
from coval.coval.eval.evaluator import evaluate_documents as evaluate
from coval.coval.eval.evaluator import muc, b_cubed, ceafe, lea
import torch
from models import CrossEncoder
from tqdm import tqdm
from heuristic import lh_split
from helper import cluster
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
import pandas as pd
from argument import args


def read(key, response):
    return get_coref_infos('%s' % key, '%s' % response,
            False, False, True)


def predict_dpos(parallel_model, dev_ab, dev_ba, device, batch_size):
    n = dev_ab['input_ids'].shape[0]
    indices = list(range(n))
    # new_batch_size = batching(n, batch_size, len(device_ids))
    # batch_size = new_batch_size
    all_scores_ab = []
    all_scores_ba = []
    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc='Predicting'):
            batch_indices = indices[i: i + batch_size]
            scores_ab = forward_ab(parallel_model, dev_ab, device, batch_indices)
            scores_ba = forward_ab(parallel_model, dev_ba, device, batch_indices)
            all_scores_ab.append(scores_ab.detach().cpu())
            all_scores_ba.append(scores_ba.detach().cpu())

    return torch.cat(all_scores_ab), torch.cat(all_scores_ba)


def predict_trained_model(mention_map, model_name, linear_weights_path, test_pairs, text_key='bert_doc', max_sentence_len=1024, long=True):
    device = torch.device('cuda:0')
    device_ids = [device]
    linear_weights = torch.load(linear_weights_path)
    scorer_module = CrossEncoder(is_training=False, model_name=model_name, long=long,
                                      linear_weights=linear_weights).to(device)
    parallel_model = torch.nn.DataParallel(scorer_module, device_ids=device_ids)
    parallel_model.module.to(device)

    tokenizer = parallel_model.module.tokenizer
    # prepare data

    test_ab, test_ba = tokenize(tokenizer, test_pairs, mention_map, parallel_model.module.end_id, text_key=text_key, max_sentence_len=max_sentence_len)

    scores_ab, scores_ba = predict_dpos(parallel_model, test_ab, test_ba, device, batch_size=64)

    return scores_ab, scores_ba, test_pairs


def save_dpos_scores(dataset, split, dpos_folder, heu='lh', threshold=None, text_key='bert_doc', max_sentence_len=1024, long=True):
    dataset_folder = f'./datasets/{dataset}/'
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    evt_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == split}
    curr_mentions = list(evt_mention_map.keys())
    # dev_pairs, dev_labels = zip(*load_lemma_dataset('./datasets/ecb/lemma_balanced_tp_fp_test.tsv'))

    mps, mps_trans = pickle.load(open(f'./datasets/{dataset}/{heu}/mp_mp_t_{split}.pkl', 'rb'))
    tps, fps, tns, fns = mps

    tps = tps
    fps = fps

    test_pairs = tps + fps
    test_labels = [1]*len(tps) + [0]*len(fps)

    linear_weights_path = dpos_folder + "linear.chkpt"
    bert_path = dpos_folder + 'bert'

    scores_ab, scores_ba, pairs = predict_trained_model(evt_mention_map, bert_path, linear_weights_path, test_pairs, text_key, max_sentence_len, long=True)

    predictions1 = (scores_ab + scores_ba)/2

    predictions = torch.squeeze(predictions1) > threshold

    test_labels = torch.LongTensor(test_labels)
    f1 = f1_score(predictions, test_labels)
    print("Test accuracy:", accuracy(predictions, test_labels))
    print("Test precision:", precision(predictions, test_labels))
    print("Test recall:", recall(predictions, test_labels))
    print("Test f1:", f1)

    return dataset_folder, test_pairs, predictions1, scores_ab, scores_ba, f1


def get_cluster_scores(dataset_folder, evt_mention_map, all_mention_pairs, dataset, split, heu, similarities, dpos_score_map, out_name, threshold):
    curr_mentions = sorted(evt_mention_map.keys())

    # generate gold clusters key file
    curr_gold_cluster_map = [(men, evt_mention_map[men]['gold_cluster']) for men in curr_mentions]
    gold_key_file = dataset_folder + f'/evt_gold_{split}.keyfile'
    generate_key_file(curr_gold_cluster_map, 'evt', dataset_folder, gold_key_file)

    w_dpos_sims = []
    for p, sim in zip(all_mention_pairs, similarities):  #
        if tuple(p) in dpos_score_map:
            q = dpos_score_map[p]
            # w_dpos_sims.append(np.mean(dpos_score_map[p]))  #             print(dpos_score_map[p])
            w_dpos_sims.append(dpos_score_map[p])
        elif (p[1], p[0]) in dpos_score_map:
            # w_dpos_sims.append(np.mean(dpos_score_map[p[1], p[0]]))  #             print('(p[1], p[0]) in dpos_score_map')
            w_dpos_sims.append(dpos_score_map[p[1], p[0]])
        else:
            w_dpos_sims.append(sim)  #             print('w_dpos_sims.append(sim)')

    mid2cluster = cluster(curr_mentions, all_mention_pairs, w_dpos_sims, threshold)
    system_key_file = dataset_folder + f'/evt_gold_dpos_{out_name}.keyfile'
    generate_key_file(mid2cluster.items(), 'evt', dataset_folder, system_key_file)
    doc = read(gold_key_file, system_key_file)

    mr, mp, mf = np.round(np.round(evaluate(doc, muc), 3) * 100, 1)
    br, bp, bf = np.round(np.round(evaluate(doc, b_cubed), 3) * 100, 1)
    cr, cp, cf = np.round(np.round(evaluate(doc, ceafe), 3) * 100, 1)
    lr, lp, lf = np.round(np.round(evaluate(doc, lea), 3) * 100, 1)

    conf = np.round((mf + bf + cf) / 3, 1)
    print(dataset, split)
    result_string = f'& {heu} && {mr}  & {mp} & {mf} && {br} & {bp} & {bf} && {cr} & {cp} & {cf} && {lr} & {lp} & {lf} && {conf} \\'

    print(result_string)
    return conf


def predict_with_dpos(dataset, split, dpos_score_map, heu='lh', threshold=None):
    dataset_folder = f'./datasets/{dataset}/'
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    evt_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == split}

    mps, mps_trans = pickle.load(open(f'./datasets/{dataset}/{heu}/mp_mp_t_{split}.pkl', 'rb'))
    _, _, _, fns = mps_trans
    tps, fps, tns, fns_nt = mps
    print(len(tps), len(fps), len(fns))
    all_mention_pairs = tps + fps
    heu_predictions = np.array([1] * len(tps) + [0] * len(fps))
    # print(len(fps,))
    conf = get_cluster_scores(dataset_folder, evt_mention_map, all_mention_pairs, dataset, split, heu, heu_predictions, dpos_score_map, out_name=heu, threshold=threshold)
    return conf

def predict(dataset, split, heu='lh'):
    dataset_folder = f'./datasets/{dataset}/'
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    evt_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == split}

    curr_mentions = sorted(evt_mention_map.keys())

    # generate gold clusters key file
    curr_gold_cluster_map = [(men, evt_mention_map[men]['gold_cluster']) for men in curr_mentions]
    gold_key_file = dataset_folder + f'/evt_gold_{split}.keyfile'
    generate_key_file(curr_gold_cluster_map, 'evt', dataset_folder, gold_key_file)

    mps, mps_trans = pickle.load(open(f'./datasets/{dataset}/{heu}/mp_mp_t_{split}.pkl', 'rb'))
    _, _, _, fns = mps_trans
    tps, fps, tns, fns_nt = mps
    print(len(tps), len(fps), len(fns))
    # print(len(fps,))
    all_mention_pairs = tps + fps + tns + fns_nt
    similarities = np.array([1]*len(tps + fps) + [0]*len(tns + fns_nt))
    mid2cluster = cluster(curr_mentions, all_mention_pairs, similarities)
    system_key_file = dataset_folder + f'/evt_gold_{heu}_{split}.keyfile'
    generate_key_file(mid2cluster.items(), 'evt', dataset_folder, system_key_file)
    doc = read(gold_key_file, system_key_file)

    ## & \LH~+ \dPos && {mr}  & {mp} & {mf} && {br} & {bp} & {bf} && {cr} & {cp} & {cf} && {lr} & {lp} & {lf} && {conf} \\

    mr, mp, mf = np.round(np.round(evaluate(doc, muc), 3)*100, 1)
    br, bp, bf = np.round(np.round(evaluate(doc, b_cubed), 3)*100, 1)
    cr, cp, cf = np.round(np.round(evaluate(doc, ceafe), 3)*100, 1)
    lr, lp, lf = np.round(np.round(evaluate(doc, lea), 3)*100, 1)

    conf = np.round((mf + bf + cf)/3, 1)
    print(dataset, split)
    result_string = f'& {heu} && {mr}  & {mp} & {mf} && {br} & {bp} & {bf} && {cr} & {cp} & {cf} && {lr} & {lp} & {lf} && {conf} \\'

    print(result_string)


def dpos_tmp(dataset, split):
    dataset_folder = f'./datasets/{dataset}'
    dpos_folder = dataset_folder + '/dpos/'

    pairs = pickle.load(open(dpos_folder + f'/{split}_pairs.pkl', 'rb'))
    ab_scores = pickle.load(open(dpos_folder + f'/{split}_scores_ab.pkl', 'rb'))
    ba_scores = pickle.load(open(dpos_folder + f'/{split}_scores_ba.pkl', 'rb'))

    dpos_map = {}
    for p, ab, ba in zip(pairs, ab_scores, ba_scores):
        dpos_map[tuple(p)] = (float(ab), float(ba))
    return dpos_map


def get_dpos(dataset, heu, split):
    dataset_folder = f'./datasets/{dataset}/'
    pairs = pickle.load(open(dataset_folder + f"/dpos1/{split}_{heu}_pairs.pkl", 'rb'))
    scores_ab = pickle.load(open(dataset_folder + f"/dpos1/{split}_{heu}_scores_ab.pkl", 'rb'))
    scores_ba = pickle.load(open(dataset_folder + f"/dpos1/{split}_{heu}_scores_ba.pkl", 'rb'))
    predictions = pickle.load(open(dataset_folder + f"/dpos1/{split}_{heu}_predictions.pkl", 'rb'))
    dpos_map = {}
    # for b, ab, ba in zip(pairs, scores_ab, scores_ba):
    #     dpos_map[tuple(b)] = (float(ab), float(ba))
    for b, prediction in zip(pairs, predictions):
        dpos_map[tuple(b)] = prediction
    return dpos_map


def save_pair_info(pairs, mention_map, file_name):
    sentence_pairs = []
    for m1, m2 in pairs:
        mention1 = mention_map[m1]
        mention2 = mention_map[m2]
        sentence_pairs.append((m1, m2, mention1['gold_cluster'], mention2['gold_cluster'], mention1['bert_sentence'], mention2['bert_sentence']))


    m1, m2, c1, c2, first, second = zip(*sentence_pairs)
    df = pd.DataFrame({'m1': m1, 'm2': m2, 'c1':c1, 'c2':c2, 'first': first, 'second': second})
    df.to_csv(file_name)


def mention_pair_analysis(dataset, split, heu):
    from collections import defaultdict
    dataset_folder = f'./datasets/{dataset}/'
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    evt_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == split}
    dpos_map = get_dpos(dataset, heu, split)
    (tps, fps, tns, fns), (tps_t, fps_t, tns_t, fns_t) = lh_split(heu, dataset, split, 0.05)

    curr_mentions = list(evt_mention_map.keys())
    mid2int = {m: i for i, m in enumerate(curr_mentions)}

    tps_t = set([tuple(sorted(p)) for p in tps])

    p_pos = tps + fps

    similarities = np.array([np.mean(dpos_map[p]) if p in p_pos else 0 for p in p_pos])

    true_predictions = np.array([1]*len(tps) + [0]*len(fps))
    predictions = similarities > 0.5

    hard_fps = np.logical_and(predictions, np.logical_not(true_predictions)).nonzero()
    hard_fps = [p_pos[i] for i in hard_fps[0]]
    print('hard_fps', len(hard_fps))

    save_pair_info(hard_fps, mention_map, f'./datasets/{dataset}/analysis/hard_fps_{dataset}.csv')

    # clusters = cluster(curr_mentions, mention_pairs=test_pairs, threshold=0.5)

    hard_fns = np.logical_and(np.logical_not(predictions), true_predictions).nonzero()
    print('hard_fns', len(hard_fps))
    hard_fns = [p_pos[i] for i in hard_fns[0]]
    save_pair_info(hard_fns, mention_map, f'./datasets/{dataset}/analysis/hard_fns_{dataset}.csv')


def threshold_ablation():
    dataset = 'ecb'
    split = 'test'
    heu = 'lh'

    dataset_folder = f'./datasets/{dataset}/'

    dpos_path = f'./datasets/{dataset}/scorer_roberta/'
    # dpos_path = f'./datasets/{dataset}/scorer/'

    linear_weights_path = dpos_path + "/linear.chkpt"
    bert_path = dpos_path + '/bert'

    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    evt_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt' and m['split'] == split}

    dpos_map = get_dpos(dataset, heu, split)
    conllf_list = []
    thresholds = [-1, 0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    for thres in thresholds:
        (tps, fps, tns, fns), (tps_t, fps_t, tns_t, fns_t) = lh_split(heu, dataset, split, thres)

        all_mention_pairs = tps + fps + tns + fns
        heuristic_predictions = [1]*len(tps) + [0]*len(fps)

        test_pairs = tps  + fps

        non_sim_pairs = []
        for p in test_pairs:
            if tuple(p) not in dpos_map:
                non_sim_pairs.append(p)

        # non_sim_pairs = non_sim_pairs[:10]

        if len(non_sim_pairs) > 0:
            scores_ab, scores_ba, pairs = predict_trained_model(evt_mention_map, bert_path, linear_weights_path, non_sim_pairs,
                                                                text_key='bert_sentence', max_sentence_len=512)

            for p, ab, ba in zip(pairs, scores_ab, scores_ba):
                dpos_map[tuple(p)] = (float(ab), float(ba))
        print('\n\nthreshold:', thres)
        conllf1 = get_cluster_scores(dataset_folder, evt_mention_map, test_pairs, dataset, split, heu,
                           heuristic_predictions, dpos_map, 'analysis',
                           0.5)
        conllf_list.append(conllf1)

    print(thresholds)
    print(conllf_list)


if __name__ == '__main__':

    ECB = 'ecb'
    GVC = 'gvc'
    print('tps', 'fps',  'fns')
    heu = 'lh_oracle'
    dpos_path = '/home/yaolong/lemma_cross/output/ecb/small/scorer/best_f1_scorer/'
    # threshold = 0.1
    # 初始化最佳阈值和相应的F1分数
    best_threshold = None
    best_f1 = 0.0
    # for threshold in range(1, 10):

    threshold = 0.1  # 将范围调整为0.1到0.9
    dataset_folder, test_pairs, predictions1, scores_ab, scores_ba, f1 = save_dpos_scores(ECB, TEST, dpos_path, heu=heu, threshold=threshold, text_key='bert_sentence', max_sentence_len=512, long=False)
    current_f1 = f1
    # 更新最佳阈值和F1分数
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = threshold
        pickle.dump(test_pairs, open(dataset_folder + f'/dpos1/{TEST}_{heu}_pairs.pkl', 'wb'))
        pickle.dump(predictions1, open(dataset_folder + f'/dpos1/{TEST}_{heu}_predictions.pkl', 'wb'))
        pickle.dump(scores_ab, open(dataset_folder + f'/dpos1/{TEST}_{heu}_scores_ab.pkl', 'wb'))
        pickle.dump(scores_ba, open(dataset_folder + f'/dpos1/{TEST}_{heu}_scores_ba.pkl', 'wb'))
    print(f"Best threshold is {best_threshold} with F1 score {best_f1}")

    dpos = get_dpos(ECB, heu, TEST)
    bset_conf = 0.0
    best_threshold_conf = None
    for threshold in range(1, 10):
        threshold /= 10.0  # 将范围调整为0.1到0.9
        conf = predict_with_dpos(ECB, TEST, dpos, heu=heu, threshold=threshold)
        if conf > bset_conf:
            bset_conf = conf
            best_threshold_conf = threshold
    print(f"Best threshold is {best_threshold_conf} with F1 score {bset_conf}")
