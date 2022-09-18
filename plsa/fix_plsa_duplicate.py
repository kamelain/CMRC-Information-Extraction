import json, collections

save_path_pref = 'dataset_plsa/quac10_v1.1/'                                        # common prefix of processed documents  
data_path = "dataset_plsa/quac10_v1/train_v0.2.json"                                # clustered source 
data_t_path = "dataset_plsa/quac10_v1/val_v0.2.json"
data_new_path = save_path_pref + "train_v0.2.json"                                  # processed dataset 
data_new_t_path = save_path_pref + "val_v0.2.json"


print(" === LOAD DATA === ")
with open(data_path, encoding = 'utf-8') as data_json, open(data_t_path, encoding = 'utf-8') as data_t_json:
    dataset = collections.OrderedDict(json.load(data_json))
    dataset_t = collections.OrderedDict(json.load(data_t_json))
new_data = dataset
new_data_t = dataset_t

for article in new_data['data']:
    for paragraph in article['paragraphs']:
        for qas in paragraph['qas']:
            pass
            