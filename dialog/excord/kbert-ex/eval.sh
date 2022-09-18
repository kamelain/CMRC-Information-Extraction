# my_INPUT_DIR=datasets/
my_INPUT_DIR=datasets/small
my_OUTPUT_DIR=output/small
# my_OUTPUT_DIR=pretrain_model/best_model/
my_batch_size=50

CUDA_VISIBLE_DEVICES=2 \
    python run_quac.py \
	--model_type roberta \
	--model_name_or_path ${my_OUTPUT_DIR} \
	--cache_prefix roberta \
	--data_dir ${my_INPUT_DIR} \
	--predict_file dev.json \
	--output_dir ${my_OUTPUT_DIR} \
	--do_eval \
	--per_gpu_eval_batch_size ${my_batch_size} \
	--threads 20 > eval_small.log

unset "${!my_@}"