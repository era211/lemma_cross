# Heuristic
from heuristic import lh, lh_oracle
from training import train_dpos
from helper import DEV, TEST
from helper import DEV, TEST
from prediction import predict
from prediction import predict_with_dpos, save_dpos_scores, get_dpos

ECB='ecb'
GVC='gvc'
# # LH - ECB
# lh(ECB, threshold=0.05)
#
# # LH - GVC
# lh(GVC, threshold=0.05)
#
# # LH_ORACLE - ECB, GVC
# lh_oracle(ECB, threshold=0.05)
# lh_oracle(GVC, threshold=0.05)


# Training
# D-Small
#  batch_size=20, n_iters=10, lr_lm(Language model)=0.000001, lr_class(classifier)=0.0001, max_sequence_length = 512
# train_dpos(ECB, model_name='/home/yaolong/PT_MODELS/PT_MODELS/roberta-base')
# train_dpos(GVC, model_name='/home/yaolong/PT_MODELS/PT_MODELS/roberta-base')
device = 5
# D-Long
#  batch_size=20, n_iters=10, lr_lm(Language model)=0.000001, lr_class(classifier)=0.0001, max_sequence_length = 1024
print('dataset: ecb')
train_dpos(ECB, model_name='/home/yaolong/PT_MODELS/PT_MODELS/allenai/longformer-base-4096', PLM='long', device=device)
print('dataset: gvc')
train_dpos(GVC, model_name='/home/yaolong/PT_MODELS/PT_MODELS/allenai/longformer-base-4096', PLM='long', device=device)

# Prediction
# Baselines
# ## LH predict ecb
# predict(ECB, TEST, heu='lh')
#
# ## LH_ORACLE predict ecb
# predict(ECB, TEST, heu='lh_oracle')
#
# ## LH predict gvc
# predict(GVC, TEST, heu='lh')
#
# ## LH_ORACLE predict gvc
# predict(GVC, TEST, heu='lh_oracle')

# Running LH + D_small
from prediction import predict_with_dpos, save_dpos_scores, get_dpos


print("运行结束！")