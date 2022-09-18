import json, collections, os.path
from progressbar import *

save_path_pref = 'dataset_plsa/quac10_v2/'                                          # common prefix of processed documents  
data_path = "dataset_plsa/tag/train_v0.2.json"                                # clustered source 
data_t_path = "dataset/quac_bg/QuAC10/val_v0.2.json"
# save_path_pref = 'dataset_plsa/quac10_v2_small/'                                  
# data_path = "dataset_plsa/quac10_v2_small_source/train_v0.2.json"                     
# data_t_path = "dataset_plsa/quac10_v2_small_source/val_v0.2.json"
data_new_path = save_path_pref + "train_v0.2.json"                                  # processed dataset 
data_new_t_path = save_path_pref + "val_v0.2.json"
q_plsa_list_path = save_path_pref + "q_plsa_list.json"

id_list_path_prefix = 'dataset_context_line/train_id/'                 # + '_id_list_' + (index//1000 +1)
id_list_path_t = 'dataset_context_line/test/_id_list.txt'       
id_topic_path_prefix = 'dataset_context_line/train_save/'              # + (index//1000 +1) + /       + id + (2)
id_topic_path_prefix_t = 'dataset_context_line/test/'

cnt_w_tag, cnt_wo_tag, cnt_bar, cnt_bar_t = 0, 0, 0, 0

print(" === LOAD DATA === ")
with open(data_path, encoding = 'utf-8') as data_json, open(data_t_path, encoding = 'utf-8') as data_t_json:
    dataset = collections.OrderedDict(json.load(data_json))
    dataset_t = collections.OrderedDict(json.load(data_t_json))
new_data = dataset
new_data_t = dataset_t
q_plsa_list = collections.OrderedDict()

print(" === INIT BLANK Q_PL === ")
for article in new_data['data']:
    for paragraph in article['paragraphs']:
        for qas in paragraph['qas']:
            if 'question_plsa' not in qas.keys():
                qas['question_plsa'] = qas['question']
                qas['question_plsa_list'] = []
for article in new_data_t['data']:
    for paragraph in article['paragraphs']:
        for qas in paragraph['qas']:
            if 'question_plsa' not in qas.keys():
                qas['question_plsa'] = qas['question']
                qas['question_plsa_list'] = []


print(" === INJECT PLSA KNOWLEDGE (TRAINING) === ")
pbar = ProgressBar().start()
for idx_article, article in enumerate(new_data['data']):
    
    # Progress Bar
    try:
        pbar.update(int((idx_article / (len(new_data['data']) - 1)) * 100))
    except:
        cnt_bar = cnt_bar + 1

    # Progress
    for idx_paragraph, paragraph in enumerate(article['paragraphs']):

        if 'tag' not in paragraph.keys(): 
            cnt_wo_tag = cnt_wo_tag + 1
            continue
        cnt_w_tag = cnt_w_tag + 1
        plsa_lists = []
        id = paragraph['id']
        id_part = paragraph['tag']

        # open topic index file
        id_list_path = id_list_path_prefix + '_id_list_' + str(id_part)
        with open(id_list_path, encoding = 'utf-8') as train_id_file:
            id_set = list(json.load(train_id_file))

        # open topic file
        
        # for id_file in id_set:
        id_topic_path = id_topic_path_prefix + str(id_part) + '/' + str(id) + '(2)' + '_topics.txt'
        if os.path.exists(id_topic_path):
            f = open(id_topic_path, "r")
            for line in f.readlines(): 
                plsa_lists.append(line.split())
            f.close()
        # else:
        #     print("plsa file not exist, id_topic_path: ", id_topic_path)

        # process each q in dataset
        for qas in paragraph['qas']:
            q_new, q_list, q_done_list = '', [], []
            q = qas['question']
            
            for wd_q in q.split():
                q_new = q_new + " " + wd_q
                for plsa_list in plsa_lists:
                    for idx, wd_pl in enumerate(plsa_list):
                        if wd_q not in q_done_list and wd_q.casefold()==wd_pl.casefold(): 
                            if idx==0: wd_kg = plsa_list[1]
                            elif idx==1: wd_kg = plsa_list[0]
                            q_new = q_new + " " + wd_kg
                            q_list.append(wd_kg)                # [basie, prominence]       -> extend kg
                            q_done_list.append(wd_q)            # ['clarinet', 'count']     -> entity

            qas["question_plsa"] = q_new
            # q_plsa_list[qas["id"]] = [q ,q_new]
            q_plsa_list[qas["id"]] = [q_list]
            qas["question_plsa_list"] = q_list
pbar.finish()

print("cnt_w_tag : ", cnt_w_tag, "; cnt_wo_tag : ", cnt_wo_tag)
print("cnt_bar", cnt_bar)

print(" === OUTPUT (TRAINING) === ")
with open(data_new_path, "w") as n_data_file:
    json.dump(new_data, n_data_file)
    

print(" === INJECT PLSA KNOWLEDGE (TESTING) === ")
pbar = ProgressBar().start()
for idx_article, article in enumerate(new_data_t['data']):

    # Progress Bar
    try:
        pbar.update(int((idx_article / (len(new_data_t['data']) - 1)) * 100))
    except:
        cnt_bar_t = cnt_bar_t + 1

    # Progress
    for idx_paragraph, paragraph in enumerate(article['paragraphs']):

        plsa_lists = []
        id = paragraph['id']
        # open topic index file

        with open(id_list_path_t, encoding = 'utf-8') as train_id_file:
            id_set = list(json.load(train_id_file))

        # open topic file
        id_topic_path = id_topic_path_prefix_t + str(id) + '(2)' + '_topics.txt'
        if os.path.exists(id_topic_path):
            f = open(id_topic_path, "r")
            for line in f.readlines(): 
                plsa_lists.append(line.split())

        # process each q in dataset
        for qas in paragraph['qas']:
            q_new, q_list, q_done_list = '', [], []
            q = qas['question']
            
            for wd_q in q.split():
                q_new = q_new + " " + wd_q
                for plsa_list in plsa_lists:
                    for idx, wd_pl in enumerate(plsa_list):
                        if wd_q not in q_done_list and wd_q.casefold()==wd_pl.casefold():
                            if idx==0: wd_kg = plsa_list[1]
                            elif idx==1: wd_kg = plsa_list[0]
                            q_new = q_new + " " + wd_kg
                            q_list.append(wd_kg)
                            q_done_list.append(wd_q)
            qas["question_plsa"] = q_new
            # q_plsa_list[qas["id"]] = [q ,q_new]
            q_plsa_list[qas["id"]] = [q_list]
            qas["question_plsa_list"] = q_list
pbar.finish()

print("cnt_bar_t", cnt_bar_t)

print(" === OUTPUT (TESTING) === ")
with open(data_new_t_path, "w") as n_data_t_file, \
    open(q_plsa_list_path, "w") as q_plsa_list_file:
    json.dump(new_data_t, n_data_t_file)
    json.dump(q_plsa_list, q_plsa_list_file)






# topics_path = 'dataset_context_line_small/C_f726b9f556564c25a23df832b054406d_1(2)_topics.txt'
# # [[basie, clarinet], [count, prominence], ...] 
# q = 'what is clarinet count ?'     
# q_new = ''                      # 'what is clarinet basie count prominence?' 
# q_list = []                     # [basie, prominence]
# q_done_list = []
# wd_lists = []
# f = open(topics_path, "r")
# for line in f.readlines(): 
#     wd_list = line.split()
#     wd_lists.append(wd_list)

# for wd_q in q.split():
#     print(wd_q)
#     q_new = q_new + " " + wd_q
#     for list in wd_lists:
#         for idx, wd_pl in enumerate(list):
#             if wd_q not in q_done_list and wd_q==wd_pl:
#                 if idx==0: wd_kg = list[1]
#                 elif idx==1: wd_kg = list[0]
#                 q_new = q_new + " " + wd_kg
#                 q_list.append(wd_kg)
#                 q_done_list.append(wd_q)
    
# print("q_new: ", q_new)
# print("q_list", q_list)
# print("q_done_list: ", q_done_list)



# path
# dataset_context_line/test/_id_list.txt                                                index   testing
# hae_kg2/plsa/dataset_context_line/test/C_0a189c86d79e42a5b87402ee6294542d_0(2)        file

# dataset_context_line/train_id/" + '_id_list_' + str(i)                                        training
# hae_kg2/plsa/dataset_context_line/train_id
# hae_kg2/plsa/dataset_context_line/train_save/1                                        file