import json, collections     

cls_type = "context20/"     # bg10, title20, context20
source_ex_pref = "../datasets/"
source_cls10_pref = cls_type  #bg10, title20, context20
save_path_pref = '../datasets_combined/' + cls_type

data1_path = source_ex_pref + "train.json"                                              # source 
data1_t_path = source_ex_pref + "dev.json"
data2_path = source_cls10_pref + "train_v0.2.json"                                             # source 
data2_t_path = source_cls10_pref + "val_v0.2.json"
data_new_path = save_path_pref + "train.json"                                       # processed dataset 
data_new_t_path = save_path_pref + "dev.json"


print("== LOAD DATA ==")
with open(data1_path, encoding = 'utf-8') as data1_json, open(data1_t_path, encoding = 'utf-8') as data1_t_json, \
    open(data2_path, encoding = 'utf-8') as data2_json, open(data2_t_path, encoding = 'utf-8') as data2_t_json:
    dataset1 = collections.OrderedDict(json.load(data1_json))
    dataset1_t = collections.OrderedDict(json.load(data1_t_json))
    dataset2 = collections.OrderedDict(json.load(data2_json))
    dataset2_t = collections.OrderedDict(json.load(data2_t_json))

new_data, new_data_t = collections.OrderedDict(), collections.OrderedDict()
new_data['data'], new_data_t['data'] = [], []

print("== RETRIEVE DATA ==")
for idx_a, article in enumerate(dataset1['data']):
    for idx_p, paragraph in enumerate(article['paragraphs']):  
        new_data['data'].append(article)
        type_label = int(dataset2['data'][idx_a]['type_label'])
        new_data['data'][idx_a].setdefault('type_label')
        new_data['data'][idx_a]['type_label'] = type_label
        
for idx_a, article in enumerate(dataset1_t['data']):
    for idx_p, paragraph in enumerate(article['paragraphs']):  
        new_data_t['data'].append(article)
        type_label = int(dataset2_t['data'][idx_a]['type_label'])
        new_data_t['data'][idx_a].setdefault('type_label')
        new_data_t['data'][idx_a]['type_label'] = type_label


print(" === OUTPUT === ")
with open(data_new_path, "w") as n_data_file, open(data_new_t_path, "w") as n_data_t_file:
    json.dump(new_data, n_data_file)
    json.dump(new_data_t, n_data_t_file)