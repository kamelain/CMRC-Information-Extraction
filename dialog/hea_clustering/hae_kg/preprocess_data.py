import json, numpy, collections, sys, jieba
import wordsegment as ws

# COMMON SETTINGS
num_clusters = 250                                                                       
cls_type = 'context'                                                                # cls_type = 'background'

source_path_pref = 'plsa-master/dataset_plsa/quac/' + cls_type + '/' + str(num_clusters) + '/'  # source path prefix
data_path = source_path_pref + 'test.json'   
data_t_path = source_path_pref + 'test.json'  
# data_path = source_path_pref + 'train_v0.2.json'                                      # source 
# data_t_path = source_path_pref + 'val_v0.2.json'
data_new_path = 'QuAC_kg/' + 'train_v0.2.json'                                      # save path
data_new_t_path = 'QuAC_kg/' + 'val_v0.2.json'

# INITIAL DS
doc_list = doc_list_t = doc_list_new = doc_list_t_new = []

# FUNCTIONS
def matched_kg(cls_num:int, match_q: str):
    # init
    path = source_path_pref + 'cls_' + str(cls_num) + '.txttopics_'  + str(cls_num) + '.txt'
    num_lines = sum(1 for line in open(path))

    # load plsa file
    with open(path, 'r') as file:           # file = codecs.open(path, 'r', 'utf-8')
        documents = [document.strip() for document in file] 
    file.close()
    
    # words segmentation
    # match_q_list = jieba.cut(match_q)     # chinese 
    ws.load()                               # english
    match_q_list = ws.segment(match_q)

    kg_data = [[None] * len(list(match_q_list)) ] * len(list(documents))  # kg_data[line#][q_wd#]: word list
    full_size = 0
    
    # match plsa file and query
    for idx, document in enumerate(documents):          # documents: a topic word file, document: a line in the file
        seg_list = jieba.cut(document)                  # seg_list: each word in the line
        for match_idx, match_wd in enumerate(match_q_list):
            for word in seg_list:
                for q_word in match_q_list: 
                    if match_wd == word:                    # if match_wd in KG 
                        if q_word != word:                  # if the KG wd not in match_q
                            kg_data[idx][match_idx].append(word)
                            full_size = full_size + 1       # KG words not in match_q
    # if full_size != 0:
    #     print("full_size", full_size)
    print("##")
    print("match_q", match_q)
    print("full_size", full_size)
    return kg_data

# LOAD DATA
print("== LOAD DATA ==")
with open(data_path, encoding = 'utf-8') as data_json, open(data_t_path, encoding = 'utf-8') as data_t_json:
    dataset = collections.OrderedDict(json.load(data_json))
    dataset_t = collections.OrderedDict(json.load(data_t_json))
new_data = dataset
new_data_t = dataset_t

for idx_0, article in enumerate(dataset['data']):
    cls_label = article['type_label']
    for idx_1, paragraph in enumerate(article['paragraphs']):
        for idx_2, qas in enumerate(paragraph['qas']):
            q = qas['question']
            q_kg = matched_kg(cls_label, q)
            new_data['data'][idx_0]['paragraphs'][idx_1]['qas'][idx_2]['kg'] = q_kg
            # if change_label:
            #     new_data['data'][idx_0]['paragraphs'][idx_1]['qas'][idx_2]['kg'] = q_kg
for idx_0, article in enumerate(dataset_t['data']):
    cls_label = article['type_label']
    for idx_1, paragraph in enumerate(article['paragraphs']):
        for idx_2, qas in enumerate(paragraph['qas']):
            q = qas['question']
            q_kg = matched_kg(cls_label, q)
            new_data_t['data'][idx_0]['paragraphs'][idx_1]['qas'][idx_2]['kg'] = q_kg

# DUMP NEW DATASETS
print("== DUMP NEW DATASET ==")
with open(data_new_path, "w") as n_data_file, open(data_new_t_path, "w") as n_data_t_file:
    json.dump(new_data, n_data_file)
    json.dump(new_data_t, n_data_t_file)
