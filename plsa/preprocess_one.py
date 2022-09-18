import json, collections, os.path

save_path_pref = 'dataset_plsa/quac10_v2_one/'                                  
data_path = "dataset_plsa/quac10_v2_small_source/one.json"                     
data_new_path = save_path_pref + "train_v0.2.json"                                  # processed dataset 
q_plsa_list_path = save_path_pref + "q_plsa_list.json"

id_list_path_prefix = 'dataset_context_line/train_id/'                 # + '_id_list_' + (index//1000 +1)
id_topic_path_prefix = 'dataset_context_line/train_save/'              # + (index//1000 +1) + /       + id + (2)

cnt_w_tag, cnt_wo_tag = 0, 0

print(" === LOAD DATA === ")
with open(data_path, encoding = 'utf-8') as data_json:
    dataset = collections.OrderedDict(json.load(data_json))
new_data = dataset
q_plsa_list = collections.OrderedDict()

print(" === INIT BLANK Q_PL === ")
for article in new_data['data']:
    for paragraph in article['paragraphs']:
        for qas in paragraph['qas']:
            if 'question_plsa' not in qas.keys():
                qas['question_plsa'] = qas['question']
                qas['question_plsa_list'] = []


print(" === INJECT PLSA KNOWLEDGE (TRAINING) === ")
for idx_article, article in enumerate(new_data['data']):
    for paragraph in article['paragraphs']:

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
        id_topic_path = id_topic_path_prefix + str(id_part) + '/' + str(id) + '(2)' + '_topics.txt'
        if os.path.exists(id_topic_path):
            f = open(id_topic_path, "r")
            for line in f.readlines(): 
                plsa_lists.append(line.split())
        else:
            print("plsa file not exist, id_topic_path: ", id_topic_path)

        print("plsa_lists: ", plsa_lists)

            # process each q in dataset
        for qas in paragraph['qas']:
            q_new, q_list, q_done_list = '', [], []
            q = qas['question']
            
            print("q: ", q)

            for wd_q in q.split():
                q_new = q_new + " " + wd_q
                for plsa_list in plsa_lists:
                    for idx, wd_pl in enumerate(plsa_list):
                        if wd_q not in q_done_list and wd_q.casefold()==wd_pl.casefold() and wd_pl not in q.split() : 
                            if idx==0: wd_kg = plsa_list[1]
                            elif idx==1: wd_kg = plsa_list[0]
                            q_new = q_new + " " + wd_kg
                            q_list.append(wd_kg)                # [basie, prominence]       -> extend kg
                            q_done_list.append(wd_q)            # ['clarinet', 'count']     -> entity

            qas["question_plsa"] = q_new
            # q_plsa_list[qas["id"]] = [q ,q_new]
            q_plsa_list[qas["id"]] = [q_list]
            qas["question_plsa_list"] = q_list
            print("q_list: ", q_list)

# print("cnt_w_tag : ", cnt_w_tag, "; cnt_wo_tag : ", cnt_wo_tag)

print(" === OUTPUT (TRAINING) === ")
with open(data_new_path, "w") as n_data_file, \
    open(q_plsa_list_path, "w") as q_plsa_list_file:
    json.dump(new_data, n_data_file)
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


dataset_plsa/quac10_v2/train_v0.2.json
dataset_plsa/quac10_v2/val_v0.2.json



# path
# dataset_context_line/test/_id_list.txt                                                index   testing
# hae_kg2/plsa/dataset_context_line/test/C_0a189c86d79e42a5b87402ee6294542d_0(2)        file

# dataset_context_line/train_id/" + '_id_list_' + str(i)                                        training
# hae_kg2/plsa/dataset_context_line/train_id
# hae_kg2/plsa/dataset_context_line/train_save/1                                        file



