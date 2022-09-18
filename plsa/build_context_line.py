import json, collections, nltk

source_path_pref = "QuAC/"
save_path_pref = "dataset_output/"

data_path = source_path_pref + "train_v0.2.json"
# data_path = source_path_pref + "train_v0.2_small.json"
data_path_t = source_path_pref + "val_v0.2.json"
train_data_path = save_path_pref + "train/"
test_data_path = save_path_pref + "test/"
train_id_path = train_data_path + "_id_list.txt"
test_id_path = test_data_path + "_id_list.txt"

common_list = ["CANNOTANSWER.", "."]

# DEFINE FUNCTION
def save_lines(save_path:str, line:str):
    f = open(save_path, 'a+')
    f.write(line)
    f.write("\n")
    f.close()

# LOAD DATA
print("# LOAD DATA")
with open(data_path, encoding = 'utf-8') as data_json, open(data_path_t, encoding = 'utf-8') as data_t_json:
    dataset = collections.OrderedDict(json.load(data_json))
    dataset_t = collections.OrderedDict(json.load(data_t_json))
# with open(data_path, encoding = 'utf-8') as data_json:
#     dataset = collections.OrderedDict(json.load(data_json))

train_id_list = []
test_id_list = []

# BUILD CONTEXT LINES FILE (TRAIN)
print("# BUILD CONTEXT LINES FILE (TRAIN)")
for article in dataset['data']:
    for paragraph in article['paragraphs']:
        text = paragraph["context"] + ". " + article["background"]
        s_list = nltk.tokenize.sent_tokenize(text)
        train_id_list.append(paragraph["id"])

        save_path = train_data_path + str(paragraph["id"]) + '.txt'
        for s in s_list:
            if s not in common_list:
                save_lines(save_path, s)

with open(train_id_path, "w") as id_list_file:
    json.dump(train_id_list, id_list_file)

# BUILD CONTEXT LINES FILE (TEST)
print("# BUILD CONTEXT LINES FILE (TEST)")
for article in dataset_t['data']:
    for paragraph in article['paragraphs']:
        text = paragraph["context"] + ". " + article["background"]
        s_list = nltk.tokenize.sent_tokenize(text)
        test_id_list.append(paragraph["id"])

        save_path = test_data_path + str(paragraph["id"]) + '.txt'
        for s in s_list:
            if s not in common_list:
                save_lines(save_path, s)

with open(test_id_path, "w") as id_list_file:
    json.dump(test_id_list, id_list_file)

# DONE
print("# DONE")