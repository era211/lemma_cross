DATE=`date '+%Y-%m-%d-%H:%M:%S'`
train=train
gpu_num=4
batch_size=12
dataset=ecb # "ecb or gvc"
PLM=small # "small or long"
exp_name=cf # "basline or cf"
save_model_path=/home/yaolong/lemma_cross/output/${exp_name}/${DATE}
output_dir=/home/yaolong/lemma_cross/output/log/${exp_name}/${PLM}
echo "Train cf-cross-encoder"
nohup python -u /home/yaolong/lemma_cross/cf_v1/c_training_v1.py --train ${train} --gpu_num ${gpu_num} --batch_size ${batch_size} --out_dir ${output_dir} --save_model_path ${save_model_path} > ${output_dir}/train_${DATE}_batch_${batch_size}.log 2>${output_dir}/train_${DATE}_batch_${batch_size}.progress &
