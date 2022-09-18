# my_INPUT_DIR=datasets_combined
my_INPUT_DIR=datasets_combined/small
my_LM=roberta
my_LM_name=roberta-base
my_OUTPUT_DIR=output/${my_LM}_small_tmp
my_batch_size=2

CUDA_VISIBLE_DEVICES=1 \
    python run_quac.py \
	--model_type ${my_LM} \
	--model_name_or_path ${my_LM_name} \
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
    > run_log_${my_LM}_small_tmp.log 

unset "${!my_@}"
