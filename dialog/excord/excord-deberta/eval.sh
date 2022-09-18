my_INPUT_DIR=datasets/
my_LM=deberta
my_LM_name=microsoft/deberta-base
my_MODEL_DIR=model/${my_LM}
my_OUTPUT_DIR=output/${my_LM}
my_batch_size=50

CUDA_VISIBLE_DEVICES=1 \
    python run_quac.py \
	--model_type ${my_LM} \
	--model_name_or_path ${my_OUTPUT_DIR} \
	--cache_prefix ${my_LM} \
	--data_dir ${my_INPUT_DIR} \
	--predict_file dev.json \
	--output_dir ${my_OUTPUT_DIR} \
	--do_eval \
	--per_gpu_eval_batch_size ${my_batch_size} \
	--threads 20 \
    > eval_log.log

unset "${!my_@}"