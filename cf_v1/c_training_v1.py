import sys
import pickle
import torch
from c_helper_v1 import tokenize, forward_ab, f1_score, accuracy, precision, recall, save_parameters, save_results_to_csv
from c_prediction_v1 import predict_dpos
import random
from tqdm import tqdm
import os
import csv
from transformers import AutoModel, AutoTokenizer
from c_models_v1 import CrossEncoder
from argument import args


def train_dpos(dataset, model_name=None, PLM=None, device=None):
    dataset_folder = f'/root/autodl-tmp/lemma_cross/datasets/{dataset}/'
    save_model_path = args.save_model_path
    mention_map = pickle.load(open(dataset_folder + "mention_map.pkl", 'rb'))
    evt_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt'}
    device = torch.device(device)
    device_ids = [device]
    # device_ids = list(range(1))
    train_mp_mpt, _ = pickle.load(open(dataset_folder + '/lh/mp_mp_t_train.pkl', 'rb'))
    dev_mp_mpt, _ = pickle.load(open(dataset_folder + '/lh/mp_mp_t_dev.pkl', 'rb'))

    tps_train, fps_train, _, _ = train_mp_mpt
    tps_dev, fps_dev, _, _ = dev_mp_mpt

    train_pairs = list(tps_train + fps_train)
    train_labels = [1]*len(tps_train) + [0]*len(fps_train)

    dev_pairs = list(tps_dev + fps_dev)
    dev_labels = [1] * len(tps_dev) + [0] * len(fps_dev)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model_name = 'roberta-base'
    scorer_module = CrossEncoder(is_training=True, tokenizer=tokenizer, model_name=model_name)
    scorer_module = scorer_module.to(device)
    parallel_model = torch.nn.DataParallel(scorer_module, device_ids=device_ids)
    parallel_model.module.to(device)

    train(dataset, train_pairs, train_labels, dev_pairs, dev_labels, parallel_model, evt_mention_map, save_model_path, device, PLM,
          batch_size=args.batch_size, n_iters=args.epoch, lr_lm=args.lr_lm, lr_class=args.lr_class)


def train(dataset,
          train_pairs,
          train_labels,
          dev_pairs,
          dev_labels,
          parallel_model,
          mention_map,
          working_folder,
          device,
          PLM,
          batch_size=8,
          n_iters=50,
          lr_lm=0.00001,
          lr_class=0.001):
    bce_loss = torch.nn.BCELoss()
    # mse_loss = torch.nn.MSELoss()
    model_params = filter(lambda p: p.requires_grad, parallel_model.module.model.parameters())
    optimizer = torch.optim.AdamW([
        {'params':model_params, 'lr': lr_lm},
        {'params': parallel_model.module.linear.parameters(), 'lr': lr_class},
        {'params': parallel_model.module.c_linear.parameters(), 'lr': lr_class},
        {'params': parallel_model.module.e_linear.parameters(), 'lr': lr_class}
    ])

    # all_examples = load_easy_hard_data(trivial_non_trivial_path)
    # train_pairs, dev_pairs, train_labels, dev_labels = split_data(all_examples, dev_ratio=dev_ratio)

    tokenizer = parallel_model.module.tokenizer

    # prepare data
    train_ab, train_ba, c_only_train_ab, c_only_train_ba, e_only_train_ab, e_only_train_ba = tokenize(tokenizer, train_pairs, mention_map, parallel_model.module.end_id, text_key='bert_sentence', max_sentence_len=args.max_sentence_len)  # 返回编码后的内容，inputs_id, attention_mask, position_id，对两个提及的句子分别进行处理，最后按行堆叠到一起
    dev_ab, dev_ba, c_only_dev_ab, c_only_dev_ba, e_only_dev_ab, e_only_dev_ba = tokenize(tokenizer, dev_pairs, mention_map, parallel_model.module.end_id, text_key='bert_sentence', max_sentence_len=args.max_sentence_len)

    # labels
    train_labels = torch.FloatTensor(train_labels)
    dev_labels = torch.LongTensor(dev_labels)
    best_f1 = 0.0
    patience = 0
    for n in tqdm(range(n_iters), desc='Epoch'):
        train_indices = list(range(len(train_pairs)))  # 得到训练对中的样本索引列表
        random.shuffle(train_indices)
        iteration_loss = 0.
        new_batch_size = batch_size
        for i in tqdm(range(0, len(train_indices), new_batch_size), desc='Training'):
            optimizer.zero_grad()
            batch_indices = train_indices[i: i + new_batch_size]

            batch_labels = train_labels[batch_indices].reshape((-1, 1)).to(device)


            f_scores_ab, c_only_scores_ab, e_only_scores_ab = forward_ab(parallel_model, train_ab, c_only_train_ab, e_only_train_ab, device, batch_indices)
            f_scores_ba, c_only_scores_ba, e_only_scores_ba = forward_ab(parallel_model, train_ba, c_only_train_ba, e_only_train_ba, device, batch_indices)
            f_scores_mean = (f_scores_ab + f_scores_ba) / 2
            c_only_scores_mean = (c_only_scores_ab + c_only_scores_ba) / 2
            e_only_scores_mean = (e_only_scores_ab + e_only_scores_ba) / 2

            full_loss = bce_loss(f_scores_mean, batch_labels)
            c_only_loss = bce_loss(c_only_scores_mean, batch_labels)
            e_only_loss = bce_loss(e_only_scores_mean, batch_labels)

            loss = full_loss + args.l_alpha*c_only_loss + args.l_beta*e_only_loss

            loss.backward()

            optimizer.step()


            iteration_loss += loss.item()

        print(f'Iteration {n} Loss:', iteration_loss / len(train_pairs))
        # iteration accuracy

        dev_scores_ab, dev_scores_ba, c_only_dev_scores_ab, c_only_dev_scores_ba, e_only_dev_scores_ab, e_only_dev_scores_ba = predict_dpos(parallel_model,
                                                                                                                                            dev_ab, dev_ba,
                                                                                                                                            c_only_dev_ab, c_only_dev_ba,
                                                                                                                                            e_only_dev_ab, e_only_dev_ba,
                                                                                                                                            device, batch_size)
        full_dev_predictions = (dev_scores_ab + dev_scores_ba)/2
        c_only_dev_predictions = (c_only_dev_scores_ab + c_only_dev_scores_ba)/2
        e_only_dev_predictions = (e_only_dev_scores_ab + e_only_dev_scores_ba)/2
        dev_predictions = full_dev_predictions - args.alpha*c_only_dev_predictions - args.beta*e_only_dev_predictions

        f_predictions = full_dev_predictions > 0.5
        f_predictions = torch.squeeze(f_predictions)
        dev_predictions = dev_predictions > 0.5
        dev_predictions = torch.squeeze(dev_predictions)

        # 事实结果
        factual_dev_accuracy = accuracy(f_predictions, dev_labels)
        factual_dev_precision = precision(f_predictions, dev_labels)
        factual_dev_recall = recall(f_predictions, dev_labels)
        factual_dev_f1 = f1_score(f_predictions, dev_labels)
        print(f"factual dev accuracy for epoch {n}:", factual_dev_accuracy)
        print(f"factual dev precision for epoch {n}:", factual_dev_precision)
        print(f"factual dev recall for epoch {n}:", factual_dev_recall)
        print(f"factual dev f1 for epoch {n}:", factual_dev_f1)

        # 反事实结果
        dev_accuracy = accuracy(dev_predictions, dev_labels)
        dev_precision = precision(dev_predictions, dev_labels)
        dev_recall = recall(dev_predictions, dev_labels)
        dev_f1 = f1_score(dev_predictions, dev_labels)
        print(f"dev accuracy for epoch {n}:", dev_accuracy)
        print(f"dev precision for epoch {n}:", dev_precision)
        print(f"dev recall for epoch {n}:", dev_recall)
        print(f"dev f1 for epoch {n}:", dev_f1)

        # 保存结果
        save_results_to_csv(n, iteration_loss / len(train_pairs), factual_dev_accuracy, factual_dev_precision, factual_dev_recall,
                            factual_dev_f1, dev_accuracy, dev_precision, dev_recall, dev_f1, working_folder, dataset, PLM)
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience = 0
            scorer_folder = working_folder + '/' + dataset + '/' + PLM + '/scorer/best_f1_scorer/'
            if not os.path.exists(scorer_folder):
                os.makedirs(scorer_folder)

            save_parameters(scorer_folder, parallel_model)

            print(f'\nsaved best f1 model\n')
        # else:
        #     patience += 1
        #     if patience > args.early_stop_patience:
        #         print("Early Stopping")
        #         sys.exit()

        if n % 2 == 0:
            scorer_folder = working_folder + '/' + dataset + '/' + PLM + f'/scorer/chk_{n}/'
            if not os.path.exists(scorer_folder):
                os.makedirs(scorer_folder)

            save_parameters(scorer_folder, parallel_model)

            print(f'saved model at {n}')


    scorer_folder = working_folder + '/' + dataset + '/' + PLM + '/scorer/final/'
    if not os.path.exists(scorer_folder):
        os.makedirs(scorer_folder)

    save_parameters(scorer_folder, parallel_model)
    print('best_f1:', best_f1)
    print(f'\nsaved final model')


if __name__ == '__main__':
    device = 0
    print(f'train  ecb ... model_name: {args.model_name}, PLM: {args.PLM}, device: {device}')
    train_dpos('ecb', model_name=args.model_name, PLM=args.PLM, device=device)
    print(f'train  gvc ... model_name: {args.model_name}, PLM: {args.PLM}, device: {device}')
    train_dpos('gvc', model_name=args.model_name, PLM=args.PLM, device=device)
