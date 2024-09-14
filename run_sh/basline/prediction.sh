DATE=`date '+%Y-%m-%d-%H:%M:%S'`
train=test  # "train or eval"
batch_size=64  # "train or eval"
dataset=ecb # "ecb or gvc"
PLM=small # "small or long"
exp_name=baseline # "baseline or cf"
output_dir=/home/yaolong/lemma_cross/output/log/${exp_name}/${train}/${dataset}/${PLM}
echo "Test cf-cross-encoder"
nohup python -u prediction.py --out_dir ${output_dir}  > ${output_dir}/test_${DATE}_batch_${batch_size}.log 2>${output_dir}/test_${DATE}_batch_${batch_size}.progress &
