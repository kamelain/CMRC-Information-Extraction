import json, collections, os 
from transformers import BertTokenizer

source_path_pref = "/share/nas167/chinyingwu/nlp/dialog/PRGC-main/data/QuAC_test/"
bert_model_dir = "/share/nas167/chinyingwu/nlp/dialog/PRGC-main/pretrain_models/bert_base_cased"
data_path = source_path_pref + "app_triples_1.json"                                      # processed dataset 
data_t_path = source_path_pref + "app_triples_1_t.json"
data_new_path = source_path_pref + "app_triples_2.json"                                      # processed dataset 
data_new_t_path = source_path_pref + "app_triples_2_t.json"


print("== LOAD DATA ==")
with open(data_path, encoding = 'utf-8') as data_json, open(data_t_path, encoding = 'utf-8') as data_t_json:
    dataset = list(json.load(data_json))
    dataset_t = list(json.load(data_t_json))
# new_data, new_data_t = collections.OrderedDict(), collections.OrderedDict()

print("== LOAD MODEL ==")
tokenizer = BertTokenizer(vocab_file=os.path.join(bert_model_dir, 'vocab.txt'), do_lower_case=False)

for data in dataset:
    text = data['text']
    text_tokens = tokenizer.tokenize(text)
    if len(text_tokens) > 512:
        text_tokens = text_tokens[:512]
    data['text_tokens'] = text_tokens

for data in dataset_t:
    text = data['text']
    text_tokens = tokenizer.tokenize(text)
    if len(text_tokens) > 512:
        text_tokens = text_tokens[:512]
    data['text_tokens'] = text_tokens

print("== OUTPUT ==")
with open(data_new_path, "w") as n_data_file, open(data_new_t_path, "w") as n_data_t_file:
    json.dump(dataset, n_data_file)
    json.dump(dataset_t, n_data_t_file)