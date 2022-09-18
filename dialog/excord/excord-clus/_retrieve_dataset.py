import json, collections     

numb = 100
numb_t = 20

source_path_pref = 'datasets_combined/'
save_path_pref = 'datasets_combined/small/'   

data_path = source_path_pref + "train.json"                                              # source 
data_t_path = source_path_pref + "dev.json"
data_new_path = save_path_pref + "train.json"                                      # processed dataset 
data_new_t_path = save_path_pref + "dev.json"

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