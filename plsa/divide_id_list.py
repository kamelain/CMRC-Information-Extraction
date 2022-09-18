import json

train_id_path = 'dataset_context_line/train/_id_list.txt'
save_path_prefix = 'dataset_context_line/train_id/'

with open(train_id_path, encoding = 'utf-8') as train_id_file:
    id_set = list(json.load(train_id_file))

id_list = []
for i, id in enumerate(id_set):
    if len(id_list)<1000:
        id_list.append(id)
    elif len(id_list)==1000:
        save_path = save_path_prefix + '_id_list' + '_' + str(i//1000)
        with open(save_path, "w") as id_list_file:
            json.dump(id_list, id_list_file)
        id_list == []
    else:
        print("id size error")