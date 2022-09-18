import json, collections     

source_path_pref = "/share/nas167/chinyingwu/nlp/dialog/hea_clustering/bert_hae-master/cls/quac_bg/QuAC10/"
save_path_pref = "/share/nas167/chinyingwu/nlp/dialog/PRGC-main/my_preprocess/QuAC/"
kg_path_pref = "/share/nas167/chinyingwu/nlp/dialog/PRGC-main/data/QuAC_test/"
data_path = source_path_pref + "train_v0.2.json"                                              # source 
data_t_path = source_path_pref + "val_v0.2.json"
data_new_path = save_path_pref + "train_v0.2.json"                                     # processed dataset 
data_new_t_path = save_path_pref + "val_v0.2.json"
# kg_path = kg_path_pref + "app_triples_3_samll_result.json"    
kg_path = kg_path_pref + "app_triples_3.json"                                      # processed dataset 
kg_t_path = kg_path_pref + "app_triples_3_t.json"                                  # processed dataset 
kg_new_path = kg_path_pref + "app_triples_4.json"                                      # processed dataset 
kg_new_t_path = kg_path_pref + "app_triples_4_t.json"

test_save_path = "/share/nas167/chinyingwu/nlp/dialog/PRGC-main/my_preprocess/kg_list.json"
test_kg_list = []

print("== LOAD DATA ==")
with open(data_path, encoding = 'utf-8') as data_json, open(data_t_path, encoding = 'utf-8') as data_t_json, \
    open(kg_path, encoding = 'utf-8') as kg_json, open(kg_t_path, encoding = 'utf-8') as kg_t_json:
    dataset = collections.OrderedDict(json.load(data_json))
    dataset_t = collections.OrderedDict(json.load(data_t_json))
    kg = list(json.load(kg_json))
    kg_t = list(json.load(kg_t_json))
new_data = collections.OrderedDict(dataset)
new_data_t = collections.OrderedDict(dataset_t)

print("== KG ALIGNMENT ==")
for item in kg:
    text_tokens = item["text_tokens"]
    item["kg"] = []
    if item["triples"] and item["triples"] != [[]]:
        for triple in item["triples"]:
            wd1_idx = triple[0][2]
            wd2_idx_s = triple[1][1]
            wd2_idx_e = triple[1][2] 
            wd1 = text_tokens[wd1_idx-1]
            wd2 = text_tokens[wd2_idx_s]
            while wd2_idx_s < wd2_idx_e-1:
                wd2_idx_s = wd2_idx_s + 1
                wd2 = wd2 + " " + text_tokens[wd2_idx_s]
            item["kg"].append([wd1, wd2, triple[2]])
            # triple[0] = ["H", 0, 2]
            # triple[1] = ["T", 11, 14]
            # triple[2] = 68
    # item["text_tokens"] = ["Anna", "Step",...]
    # item["triples"] = [[["H", 0, 2], ["T", 11, 14], 68]]
    # item["triples"] = [wd1, wd2, 68]]
# print(kg)

for item in kg_t:
    text_tokens = item["text_tokens"]
    item["kg"] = []
    if item["triples"] and item["triples"] != [[]]:
        for triple in item["triples"]:
            wd1_idx = triple[0][2]
            wd2_idx_s = triple[1][1]
            wd2_idx_e = triple[1][2] 
            wd1 = text_tokens[wd1_idx-1]
            wd2 = text_tokens[wd2_idx_s]
            while wd2_idx_s < wd2_idx_e-1:
                wd2_idx_s = wd2_idx_s + 1
                wd2 = wd2 + " " + text_tokens[wd2_idx_s]
            item["kg"].append([wd1, wd2, triple[2]])


print("== INIT BLANK Q_PRGC ==")
for article in new_data['data']:
    for paragraph in article['paragraphs']:
        for qas in paragraph['qas']:
            if 'question_prgc' not in qas.keys():
                qas['question_prgc'] = qas['question']
                qas['question_prgc_list'] = []
for article in new_data_t['data']:
    for paragraph in article['paragraphs']:
        for qas in paragraph['qas']:
            if 'question_prgc' not in qas.keys():
                qas['question_prgc'] = qas['question']
                qas['question_prgc_list'] = []


print("== RETRIEVE DATA ==")
# new_data['data'], new_data_t['data'] = [], []
for article in new_data['data']:
    for paragraph in article['paragraphs']:  
        id = paragraph['id']
        for item in kg:
            if type(item) is dict:
                id_kg = item['id']
                # print("type(item)", type(item))
                # print("type(id)", type(id)," ,id: ", id)
                # print("type(id_kg)", type(id_kg), ", id_kg: ", id_kg)
                if id_kg==id:
                    if item["kg"]:              # kg not empty
                        for kg in item["kg"]: 
                            kg_wd1 = kg[0]
                            kg_wd2 = kg[1]
                            for qas in paragraph['qas']:
                                item_q = qas['question']
                                done_q = ""
                                kg_list = []
                                for q_wd in item_q.split():
                                    done_q = done_q + " " + q_wd
                                    if kg_wd1==q_wd:
                                        done_q = done_q + " " + kg_wd2
                                        kg_list.append(kg_wd2)
                                    elif kg_wd2==q_wd:
                                        done_q = done_q + " " + kg_wd1
                                        kg_list.append(kg_wd1)
                                qas['question_prgc'] = done_q
                                qas['question_prgc_list'] = kg_list
                                test_kg_list.append(kg_list)
                    
                    # item_q = qas['question']
                    # "question_wn": "what happened occur in 1983?", 
                    # "question_wn_list": ["occur"]


for article in new_data_t['data']:
    for paragraph in article['paragraphs']:  
        id = paragraph['id']
        for item in kg_t:
            if type(item) is dict:
                id_kg = item['id']
                if id_kg==id:
                    if item["kg"]:              # kg not empty
                        for kg in item["kg"]: 
                            kg_wd1 = kg[0]
                            kg_wd2 = kg[1]
                            for qas in paragraph['qas']:
                                item_q = qas['question']
                                done_q = ""
                                kg_list = []
                                for q_wd in item_q.split():
                                    done_q = done_q + " " + q_wd
                                    if kg_wd1==q_wd:
                                        done_q = done_q + " " + kg_wd2
                                        kg_list.append(kg_wd2)
                                    elif kg_wd2==q_wd:
                                        done_q = done_q + " " + kg_wd1
                                        kg_list.append(kg_wd1)
                                qas['question_prgc'] = done_q
                                qas['question_prgc_list'] = kg_list
                                test_kg_list.append(kg_list)


print("== OUTPUT ==")
with open(data_new_path, "w") as n_data_file, open(data_new_t_path, "w") as n_data_t_file, \
    open(kg_new_path, "w") as kg_new_file, open(kg_new_t_path, "w") as kg_new_t_file, \
    open(test_save_path, "w") as test_file:
    json.dump(new_data, n_data_file)
    json.dump(new_data_t, n_data_t_file)
    json.dump(kg, kg_new_file)
    json.dump(kg_t, kg_new_t_file)
    json.dump(test_kg_list, test_file)
