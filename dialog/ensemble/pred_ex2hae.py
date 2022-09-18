# excord pred
    # "C_5ab583f64dbb47b995cf59328ea0af43_1_q#0": "In May 1983, she married Nikos Karvelas,"
# hae pred
    # {   "best_span_str": ["In May 1983, she married Nikos Karvelas, a composer, with whom she collaborated in 1975", "she gave birth to her daughter Sofia.", "CANNOTANSWER", "After their marriage, she started a close collaboration with Karvelas.", "CANNOTANSWER", "In 1986, she participated at the Cypriot National Final for Eurovision Song Contest with the song Thelo Na Gino Star (\"I Want To Be A Star\"), taking second place.", "reached gold status selling 80.000 units.", "In 1986 I Epomeni Kinisi (\"The Next Move\") was released."], 
    #     "qid": ["C_5ab583f64dbb47b995cf59328ea0af43_1_q#0", "C_5ab583f64dbb47b995cf59328ea0af43_1_q#1", "C_5ab583f64dbb47b995cf59328ea0af43_1_q#2", "C_5ab583f64dbb47b995cf59328ea0af43_1_q#3", "C_5ab583f64dbb47b995cf59328ea0af43_1_q#4", "C_5ab583f64dbb47b995cf59328ea0af43_1_q#5", "C_5ab583f64dbb47b995cf59328ea0af43_1_q#6", "C_5ab583f64dbb47b995cf59328ea0af43_1_q#7"], 
    #     "followup": ["x", "x", "x", "x", "x", "x", "x", "x"], 
    #     "yesno": ["y", "y", "y", "y", "y", "y", "y", "y"]}

import json,collections
train_id_list_path = "/share/nas167/chinyingwu/nlp/dialog/hea_clustering/hae_kg2/plsa/dataset_context_line/train/_id_list.txt"
test_id_list_path = "/share/nas167/chinyingwu/nlp/dialog/hea_clustering/hae_kg2/plsa/dataset_context_line/test/_id_list.txt"
ori_path = "/share/nas167/chinyingwu/nlp/dialog/excord/excord-ori/excord-main/pretrain_model/best_model/predictions_.json"
result_path = "ex2hae/best.json"

# current_id = 0
# current_dict = {}
# current_dict["best_span_str"] = []
# current_dict["qid"] = []
# current_dict["followup"] = []
# current_dict["yesno"] = []
id_dict = collections.OrderedDict()
# a.split("_q#")[0]
# for line in open(ori_path, 'r'):
#     if line.strip():

def new_dict():
    current_dict = {}
    current_dict["best_span_str"] = []
    current_dict["qid"] = []
    current_dict["followup"] = []
    current_dict["yesno"] = []

def save_dict():
    with open(result_path, 'a') as f:
        json.dump(current_dict, f)
        f.write('\n')
    new_dict()

print("== LOAD DATA ==")
with open(ori_path, encoding = 'utf-8') as ori_json:
    dataset = collections.OrderedDict(json.load(ori_json))

with open(test_id_list_path, encoding = 'utf-8') as idlist_json:
    id_list = list(json.load(idlist_json))

check_id = ""
for id in id_list:
    for id_2 in dataset.keys():
        id_2_pref = id_2.split("_q#")[0]
        if id_2_pref==id: 
            if id not in id_dict.keys():
                id_dict[id] = []
            id_dict[id].append(id_2)
    check_id = id_2_pref
print("id_dict[id_pref]: ", id_dict[check_id])

for k in id_dict.keys():
    current_dict = {}
    current_dict["qid"] = id_dict[k]
    # current_dict["followup"] = ["x"]*len(id_dict)
    # current_dict["yesno"] = ["y"]*len(id_dict)
    for i in id_dict[k]:
        current_dict["best_span_str"], current_dict["followup"], current_dict["yesno"] = [], [], []
        current_dict["best_span_str"].append(dataset[i])
        current_dict["followup"].append("x")
        current_dict["yesno"].append("y")
    save_dict()





# for k in dataset.keys():
#     value = dataset[k]
#     id = k.split("_q#")[0]
#     if id!=current_id:
#         new_dict()
#     if k not in current_dict["qid"]:
#         current_dict["best_span_str"].append(value)
#         current_dict["qid"].append(k)
#         current_dict["followup"].append("x")
#         current_dict["yesno"].append("y")
#         save_dict()
#     current_id = id