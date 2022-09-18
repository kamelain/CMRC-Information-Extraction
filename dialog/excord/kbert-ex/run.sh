# my_INPUT_DIR=datasets
my_INPUT_DIR=datasets/small
my_LM=roberta
my_OUTPUT_DIR=output/small
my_batch_size=2
# my_model_name_or_path=pretrain_model/best_model/
my_model_name_or_path=roberta-base

CUDA_VISIBLE_DEVICES=1 \
    python run_quac.py \
	--model_type roberta \
	--model_name_or_path roberta-base \
	--do_train \
	--data_dir ${my_INPUT_DIR} \
	--train_file train.json \
	--output_dir ${my_OUTPUT_DIR} \
	--per_gpu_train_batch_size ${my_batch_size} \
	--num_train_epochs 2 \
	--learning_rate 3e-5 \
	--weight_decay 0.01 \
	--threads 20 \
	--excord_cons_coeff 0.5 \
	--excord_softmax_temp 1 \
    > run_log_small.log 

unset "${!my_@}"
