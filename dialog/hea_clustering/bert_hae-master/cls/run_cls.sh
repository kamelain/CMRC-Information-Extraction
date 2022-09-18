my_cls_type=title #bg, context, title
my_cls_numb=40
my_dataset_dir=quac_${my_cls_type}/QuAC${my_cls_numb}
# my_result_dir=quac_${my_cls_type}/result
my_result_dir=redo_${my_cls_type}

if [ ! -d ${my_result_dir} ]
then
    mkdir -p ${my_result_dir}
fi

CUDA_VISIBLE_DEVICES=1bash  PYTHONIOENCODING=utf-8 \
    python context_cls.py \
    ${my_cls_numb} \
    ${my_cls_type} \
    ${my_dataset_dir}/train_v0.2.json \
    ${my_dataset_dir}/val_v0.2.json \
    ${my_result_dir}/emb${my_cls_numb}.json \
    ${my_result_dir}/count${my_cls_numb}.txt \
    ${my_result_dir}/fig${my_cls_numb}.png \
    ${my_result_dir}/fig${my_cls_numb}_cnt.png \
    > ${my_result_dir}/${my_cls_type}_${my_cls_numb}.log

unset "${!my_@}"

# num_clusters = int(sys.argv[1])     
# clu_type = sys.argv[2]        context/bg/title
# data_new_path = sys.argv[3]
# data_new_t_path = sys.argv[4]
# emb_path = sys.argv[5]
# cnt_path = sys.argv[6]
# fig_path = sys.argv[7]