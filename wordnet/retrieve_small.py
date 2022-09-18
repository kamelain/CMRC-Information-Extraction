import json, collections     

numb = 100
numb_t = 20

source_path_pref = 'dataset_wn/quac10_v4/'
# save_path_pref = 'dataset_wn/quac10_v4/small/'   

# /share/nas167/chinyingwu/nlp/dialog/hea_clustering/hae_kg2/plsa/dataset_plsa/tag/train_v0.2.json

data_path = "/share/nas167/chinyingwu/nlp/dialog/hea_clustering/hae_kg2/plsa/dataset_plsa/tag/train_v0.2.json"
data_t_path = "/share/nas167/chinyingwu/nlp/dialog/hea_clustering/hae_kg2/plsa/dataset/quac_bg/QuAC10/val_v0.2.json"
save_path_pref = "/share/nas167/chinyingwu/nlp/dialog/hea_clustering/hae_kg2/plsa/dataset_plsa/quac10_v2_small_source/"

# data_path = source_path_pref + "train_v0.2.json"                                              # source 
# data_t_path = source_path_pref + "val_v0.2.json"
data_new_path = save_path_pref + "train_v0.2.json"                                      # processed dataset 
data_new_t_path = save_path_pref + "val_v0.2.json"

print("== LOAD DATA ==")
with open(data_path, encoding = 'utf-8') as data_json, open(data_t_path, encoding = 'utf-8') as data_t_json:
    dataset = collections.OrderedDict(json.load(data_json))
    dataset_t = collections.OrderedDict(json.load(data_t_json))
new_data, new_data_t = collections.OrderedDict(), collections.OrderedDict()

print("== RETRIEVE DATA ==")
new_data['data'], new_data_t['data'] = [], []
for article in dataset['data']:
    for paragraph in article['paragraphs']:  
        if len(new_data['data'])<numb:
            new_data['data'].append(article)  
for article in dataset_t['data']:
    for paragraph in article['paragraphs']:  
        if len(new_data_t['data'])<numb_t:
            new_data_t['data'].append(article)

print(" === OUTPUT === ")
with open(data_new_path, "w") as n_data_file, open(data_new_t_path, "w") as n_data_t_file:
    json.dump(new_data, n_data_file)
    json.dump(new_data_t, n_data_t_file)