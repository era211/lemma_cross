DATE=`date '+%Y-%m-%d-%H:%M:%S'`

echo "Train cf-cross-encoder"
PLM = small # small or long
exp_name = cf # basline or cf
nohup python -u c_training_v1.py > /home/yaolong/lemma_cross/output/log/${exp_name}/${PLM}/train_batch8.log 2>/home/yaolong/lemma_cross/output/log/${exp_name}/${PLM}/train_batch8.progress &
