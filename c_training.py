import pickle
import torch
import argparse
from c_helper import tokenize, forward_ab, c_only_forward_ab, e_only_forward_ab, f1_score, accuracy, precision, recall
from c_prediction import predict_dpos
import random
from tqdm import tqdm
import os
from transformers import AutoModel, AutoTokenizer
from c_models import CrossEncoder, COnlyCrossEncoder, EOnlyCrossEncoder

'''添加参数'''
parser = argparse.ArgumentParser(description='Training a Counterfactual-ECR')
parser.add_argument('--gpu_num', type=int, default=0, help=' A single GPU number')
parser.add_argument('--model_name', type=str, default='/root/lanyun-tmp/roberta-base', help='roberta-base')
parser.add_argument('--PLM', type=str, default='small', help='small or long')
parser.add_argument('--full', type=bool, default=True, help='event and context')
parser.add_argument('--c_only', type=bool, default=True, help='only context')
parser.add_argument('--e_only', type=bool, default=True, help='only event')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--epoch', default=10, type=int, help='epoch')
parser.add_argument('--lr_lm', default=0.000001, type=float, help='learning rate')
parser.add_argument('--lr_class', default=0.0001, type=float, help='linear_learning rate')
parser.add_argument('--l_alpha', default=0.25, type=float)
parser.add_argument('--l_beta', default=0.25, type=float)
parser.add_argument('--alpha', default=0.15, type=float)
parser.add_argument('--beta', default=0.15, type=float)
args = parser.parse_args()

def train_dpos(dataset, model_name=None, PLM=None, device=None):
    dataset_folder = f'./datasets/{dataset}/'
    mention_map = pickle.load(open(dataset_folder + "/mention_map.pkl", 'rb'))
    evt_mention_map = {m_id: m for m_id, m in mention_map.items() if m['men_type'] == 'evt'}
    device = torch.device(device)
    device_ids = [0, 1]
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
    model = AutoModel.from_pretrained(model_name)
    # model_name = 'roberta-base'
    full_scorer_module = CrossEncoder(is_training=True, tokenizer=tokenizer, model=model).to(device)
    c_only_scorer_module = COnlyCrossEncoder(is_training=True, tokenizer=tokenizer, model=model).to(device)
    e_only_scorer_module = EOnlyCrossEncoder(is_training=True, tokenizer=tokenizer, model=model).to(device)

    full_parallel_model = torch.nn.DataParallel(full_scorer_module, device_ids=device_ids)
    full_parallel_model.module.to(device)

    c_only_parallel_model = torch.nn.DataParallel(c_only_scorer_module, device_ids=device_ids)
    c_only_parallel_model.module.to(device)

    e_only_parallel_model = torch.nn.DataParallel(e_only_scorer_module, device_ids=device_ids)
    e_only_parallel_model.module.to(device)

    train(train_pairs, train_labels, dev_pairs, dev_labels, full_parallel_model, c_only_parallel_model, e_only_parallel_model, evt_mention_map, dataset_folder, device, PLM,
          batch_size=args.batch_size, n_iters=args.epoch, lr_lm=args.lr_lm, lr_class=args.lr_class)


def train(train_pairs,
          train_labels,
          dev_pairs,
          dev_labels,
          parallel_model,
          c_only_parallel_model,
          e_only_parallel_model,
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

    optimizer = torch.optim.AdamW([
        {'params': parallel_model.module.model.parameters(), 'lr': lr_lm},
        {'params': parallel_model.module.linear.parameters(), 'lr': lr_class},
        {'params': c_only_parallel_model.module.linear.parameters(), 'lr': lr_class},
        {'params': e_only_parallel_model.module.linear.parameters(), 'lr': lr_class}
    ])

    # all_examples = load_easy_hard_data(trivial_non_trivial_path)
    # train_pairs, dev_pairs, train_labels, dev_labels = split_data(all_examples, dev_ratio=dev_ratio)

    tokenizer = parallel_model.module.tokenizer

    # prepare data
    train_ab, train_ba, c_only_train_ab, c_only_train_ba, e_only_train_ab, e_only_train_ba = tokenize(tokenizer, train_pairs, mention_map, parallel_model.module.end_id, text_key='bert_sentence', max_sentence_len=512)  # 返回编码后的内容，inputs_id, attention_mask, position_id，对两个提及的句子分别进行处理，最后按行堆叠到一起
    dev_ab, dev_ba, c_only_dev_ab, c_only_dev_ba, e_only_dev_ab, e_only_dev_ba = tokenize(tokenizer, dev_pairs, mention_map, parallel_model.module.end_id, text_key='bert_sentence', max_sentence_len=512)

    # labels
    train_labels = torch.FloatTensor(train_labels)
    dev_labels = torch.LongTensor(dev_labels)
    f1 = 0.0
    for n in tqdm(range(n_iters), desc='Epoch'):
        train_indices = list(range(len(train_pairs)))  # 得到训练对中的样本索引列表
        random.shuffle(train_indices)
        iteration_loss = 0.
        # new_batch_size = batching(len(train_indices), batch_size, len(device_ids))
        new_batch_size = batch_size
        for i in tqdm(range(0, len(train_indices), new_batch_size), desc='Training'):
            optimizer.zero_grad()
            batch_indices = train_indices[i: i + new_batch_size]

            batch_labels = train_labels[batch_indices].reshape((-1, 1)).to(device)

            if args.full:
                scores_ab = forward_ab(parallel_model, train_ab, device, batch_indices)
                scores_ba = forward_ab(parallel_model, train_ba, device, batch_indices)
                scores_mean = (scores_ab + scores_ba) / 2
                full_loss = bce_loss(scores_mean, batch_labels)
            else:
                full_loss = 0.0

            if args.c_only:
                c_only_scores_ab = c_only_forward_ab(c_only_parallel_model, c_only_train_ab, device, batch_indices)
                c_only_scores_ba = c_only_forward_ab(c_only_parallel_model, c_only_train_ba, device, batch_indices)
                c_only_scores_mean = (c_only_scores_ab + c_only_scores_ba) / 2
                c_only_loss = bce_loss(c_only_scores_mean, batch_labels)
            else:
                c_only_loss = 0.0

            if args.e_only:
                e_only_scores_ab = e_only_forward_ab(e_only_parallel_model, e_only_train_ab, device, batch_indices)
                e_only_scores_ba = e_only_forward_ab(e_only_parallel_model, e_only_train_ba, device, batch_indices)
                e_only_scores_mean = (e_only_scores_ab + e_only_scores_ba) / 2
                e_only_loss = bce_loss(e_only_scores_mean, batch_labels)
            else:
                e_only_loss = 0.0


            loss = full_loss + args.l_alpha*c_only_loss + args.l_beta*e_only_loss

            loss.backward()

            optimizer.step()

            iteration_loss += loss.item()

        print(f'Iteration {n} Loss:', iteration_loss / len(train_pairs))
        # iteration accuracy
        dev_scores_ab, dev_scores_ba, c_only_dev_scores_ab, c_only_dev_scores_ba, e_only_dev_scores_ab, e_only_dev_scores_ba = predict_dpos(parallel_model,
                                                                                                                                            c_only_parallel_model,
                                                                                                                                            e_only_parallel_model,
                                                                                                                                            dev_ab, dev_ba,
                                                                                                                                            c_only_dev_ab, c_only_dev_ba,
                                                                                                                                            e_only_dev_ab, e_only_dev_ba,
                                                                                                                                            device, batch_size)
        full_dev_predictions = (dev_scores_ab + dev_scores_ba)/2
        c_only_dev_predictions = (c_only_dev_scores_ab + c_only_dev_scores_ba)/2
        e_only_dev_predictions = (e_only_dev_scores_ab + e_only_dev_scores_ba)/2
        dev_predictions = full_dev_predictions - args.alpha*c_only_dev_predictions - args.beta*e_only_dev_predictions

        dev_predictions = dev_predictions > 0.5
        dev_predictions = torch.squeeze(dev_predictions)

        print("dev accuracy:", accuracy(dev_predictions, dev_labels))
        print("dev precision:", precision(dev_predictions, dev_labels))
        print("dev recall:", recall(dev_predictions, dev_labels))
        print("dev f1:", f1_score(dev_predictions, dev_labels))

        dev_f1 = f1_score(dev_predictions, dev_labels)
        if dev_f1 > f1:
            f1 = dev_f1
            scorer_folder = working_folder + PLM + '/best_f1_scorer/'
            if not os.path.exists(scorer_folder):
                os.makedirs(scorer_folder)
            model_path = scorer_folder + '/linear.chkpt'
            torch.save(parallel_model.module.linear.state_dict(), model_path)
            parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
            parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')
            print(f'saved best f1 model')

        if n % 2 == 0:
            scorer_folder = working_folder + PLM + f'/scorer/chk_{n}'
            if not os.path.exists(scorer_folder):
                os.makedirs(scorer_folder)
            model_path = scorer_folder + '/linear.chkpt'
            torch.save(parallel_model.module.linear.state_dict(), model_path)
            parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
            parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')
            print(f'saved model at {n}')

    scorer_folder = working_folder + PLM + '/scorer/'
    if not os.path.exists(scorer_folder):
        os.makedirs(scorer_folder)
    model_path = scorer_folder + '/linear.chkpt'
    torch.save(parallel_model.module.linear.state_dict(), model_path)
    parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
    parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')


if __name__ == '__main__':
    device = args.gpu_num
    train_dpos('ecb', model_name=args.model_name, PLM=args.PLM, device=device)
    train_dpos('gvc', model_name=args.model_name, PLM=args.PLM, device=device)
