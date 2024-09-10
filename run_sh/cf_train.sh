DATE=`date '+%Y-%m-%d-%H:%M:%S'`
gpu_num=1
batch_size=6
dataset=ecb # "ecb or gvc"
PLM=small # "small or long"
exp_name=cf # "basline or cf"
save_model_path=/home/yaolong/lemma_cross/output/${dataset}/${exp_name}/${PLM}
output_dir=/home/yaolong/lemma_cross/output/log/${exp_name}/${PLM}
echo "Train cf-cross-encoder"
nohup python -u c_training_v1.py --gpu_num ${gpu_num} --batch_size ${batch_size} --out_dir ${output_dir} --save_model_path ${save_model_path}\
                              > ${output_dir}/train_${DATE}_batch_${batch_size}.log 2>${output_dir}/train_${DATE}_batch_${batch_size}.progress &
