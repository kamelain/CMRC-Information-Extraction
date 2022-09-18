my_INPUT_DIR=datasets
my_LM=deberta
my_LM_name=microsoft/deberta-base
my_OUTPUT_DIR=output/${my_LM}3
my_batch_size=6

if [ ! -d ${my_OUTPUT_DIR} ]
then
    mkdir ${my_OUTPUT_DIR}
fi


CUDA_VISIBLE_DEVICES=3 \
    python run_quac.py \
	--model_type ${my_LM} \
	--model_name_or_path ${my_LM_name} \
	--data_dir ${my_INPUT_DIR} \
	--train_file train.json \
	--output_dir ${my_OUTPUT_DIR} \
	--per_gpu_train_batch_size ${my_batch_size} \
	--num_train_epochs 2 \
	--learning_rate 3e-5 \
	--weight_decay 0.01 \
	--threads 20 \
	--do_train \
	--excord_cons_coeff 0.5 \
	--excord_softmax_temp 1 \
    > log/run_${my_LM}.log 

	# --do_train \

unset "${!my_@}"
