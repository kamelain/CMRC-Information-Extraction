import json, collections     

source_path_pref = "/share/nas167/chinyingwu/nlp/dialog/hea_clustering/bert_hae-master/cls/quac_bg/QuAC10/"
save_path_pref = "/share/nas167/chinyingwu/nlp/dialog/PRGC-main/data/QuAC_test/"

data_path = source_path_pref + "train_v0.2.json"                                              # source 
data_t_path = source_path_pref + "val_v0.2.json"
data_new_path = save_path_pref + "app_triples.json"                                      # processed dataset 
data_new_t_path = save_path_pref + "app_triples_t.json"

print("== LOAD DATA ==")
with open(data_path, encoding = 'utf-8') as data_json, open(data_t_path, encoding = 'utf-8') as data_t_json:
    dataset = collections.OrderedDict(json.load(data_json))
    dataset_t = collections.OrderedDict(json.load(data_t_json))
new_data, new_data_t = [], []

print("== RETRIEVE DATA ==")
for article in dataset['data']:
    a = dict()
    a['text'] = article['background']
    for paragraph in article['paragraphs']:  
        a['id'] = paragraph['id']
    new_data.append(a)

for article in dataset_t['data']:
    a = dict()
    a['text'] = article['background']
    for paragraph in article['paragraphs']:  
        a['id'] = paragraph['id']
    new_data_t.append(a)


print(" === OUTPUT === ")
with open(data_new_path, "w") as n_data_file, open(data_new_t_path, "w") as n_data_t_file:
    json.dump(new_data, n_data_file)
    json.dump(new_data_t, n_data_t_file)