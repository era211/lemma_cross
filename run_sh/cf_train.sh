DATE=`date '+%Y-%m-%d-%H:%M:%S'`
gpu_num=1
batch_size=9
PLM=small # "small or long"
exp_name=cf # "basline or cf"
output_dir=/home/yaolong/lemma_cross/output/log/${exp_name}/${PLM}
echo "Train cf-cross-encoder"
nohup python -u c_training_v1.py --gpu_num ${gpu_num} --batch_size ${batch_size} --out_dir ${output_dir}\
                              > ${output_dir}/train_batch${batch_size}.log 2>${output_dir}/train_batch${batch_size}.progress &
