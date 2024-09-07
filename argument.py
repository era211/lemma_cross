import argparse

'''添加参数'''
parser = argparse.ArgumentParser(description='Training a Counterfactual-ECR')
parser.add_argument('--gpu_num', type=int, default=4, help=' A single GPU number')
parser.add_argument('--model_name', type=str, default='/home/yaolong/PT_MODELS/PT_MODELS/roberta-base', help='roberta-base')
parser.add_argument('--PLM', type=str, default='small', help='small or long')
parser.add_argument('--full', type=bool, default=True, help='event and context')
parser.add_argument('--c_only', type=bool, default=True, help='only context')
parser.add_argument('--e_only', type=bool, default=True, help='only event')
parser.add_argument('--batch_size', default=6, type=int, help='batch size')
parser.add_argument('--epoch', default=10, type=int, help='epoch')
parser.add_argument('--lr_lm', default=0.000001, type=float, help='learning rate')
parser.add_argument('--lr_class', default=0.0001, type=float, help='linear_learning rate')
parser.add_argument('--l_alpha', default=0.25, type=float)
parser.add_argument('--l_beta', default=0.25, type=float)
parser.add_argument('--alpha', default=0.15, type=float)
parser.add_argument('--beta', default=0.15, type=float)
args = parser.parse_args()