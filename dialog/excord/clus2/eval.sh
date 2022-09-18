# my_INPUT_DIR=datasets/
imy_INPUT_DIR=datasets_combined/bg10
# my_INPUT_DIR=datasets_combined/context20
my_OUTPUT_DIR=output/v11
# my_OUTPUT_DIR=pretrain_model/best_model/
my_batch_size=50

CUDA_VISIBLE_DEVICES=1 \
    python run_quac.py \
	--model_type roberta \
	--model_name_or_path ${my_OUTPUT_DIR} \
	--cache_prefix roberta \
	--data_dir ${my_INPUT_DIR} \
	--predict_file dev.json \
	--output_dir ${my_OUTPUT_DIR} \
	--do_eval \
	--per_gpu_eval_batch_size ${my_batch_size} \
	--threads 20 > eval_v1.log

unset "${!my_@}"
