# 修 run_wn.sh 造成的 'question_wn' 'question_wn_list' 欄位空白問題

import json, numpy, collections

source_path_pref = 'dataset_wn/quac10_v2/'
save_path_pref = 'dataset_wn/quac10_v2/'        
data_path = source_path_pref + "train_v0.2.json"  
data_t_path = source_path_pref + "val_v0.2.json"
data_new_path = save_path_pref + "train_v0.2.json"           
data_new_t_path = save_path_pref + "val_v0.2.json"
kw_cnt_path = save_path_pref + "wn_cnt.json"
kw_wd_path = save_path_pref + "wn_wd.json"

print(" === LOAD DATA === ")
with open(data_path, encoding = 'utf-8') as data_json, open(data_t_path, encoding = 'utf-8') as data_t_json:
    dataset = collections.OrderedDict(json.load(data_json))
    dataset_t = collections.OrderedDict(json.load(data_t_json))
new_data = dataset
new_data_t = dataset_t

kw_cnt = {}

print(" === CHECK KG (TRAINING) === ")
for article in new_data['data']:
    for paragraph in article['paragraphs']:
        for qas in paragraph['qas']:
            for wd in qas['question_wn_list']:
                if wd not in kw_cnt.keys(): 
                    kw_cnt[wd] = 1
                else:
                    kw_cnt[wd] = kw_cnt[wd] + 1

print(" === CHECK KG (TESTING) === ")
for article in new_data_t['data']:
    for paragraph in article['paragraphs']:
        for qas in paragraph['qas']:
            for wd in qas['question_wn_list']:
                if wd not in kw_cnt.keys(): 
                    kw_cnt[wd] = 1
                else:
                    kw_cnt[wd] = kw_cnt[wd] + 1
                    
kw_cnt = sorted(kw_cnt.items(), key=lambda x: x[1], reverse=True)
kw_wd = [r[0] for r in kw_cnt]

with open(kw_cnt_path, "w") as kw_cnt_file, open(kw_wd_path, "w") as kw_wd_file:
    json.dump(kw_cnt, kw_cnt_file)
    json.dump(kw_wd, kw_wd_file)





# + blank "question_wn" "question_wn_list"

# print(" === CHECK KG (TRAINING) === ")
# for article in new_data['data']:
#     for paragraph in article['paragraphs']:
#         for qas in paragraph['qas']:
#             if 'question_wn' not in qas.keys():
#                 qas['question_wn'] = qas['question']
#                 qas['question_wn_list'] = []

# print(" === CHECK KG (TESTING) === ")
# for article in new_data_t['data']:
#     for paragraph in article['paragraphs']:
#         for qas in paragraph['qas']:
#             if 'question_wn' not in qas.keys():
#                 qas['question_wn'] = qas['question']
#                 qas['question_wn_list'] = []

# print(" === OUTPUT === ")
# with open(data_new_path, "w") as n_data_file, \
#     open(data_new_t_path, "w") as n_data_t_file:
#     json.dump(new_data, n_data_file)
#     json.dump(new_data_t, n_data_t_file)