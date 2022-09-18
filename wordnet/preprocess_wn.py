import enum
import json, numpy, collections, nltk
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

num_clusters = 20                                                                       
cls_type = 'context' 
save_path_pref = 'dataset_wd/quac/'        # common prefix of processed documents
source_path_pref = 'dataset_plsa/quac/' + cls_type + '/' + str(num_clusters) + '/'      # source path prefix (cls documents)

# dataset/quac_context/QuAC20
data_path = "dataset/quac_context/QuAC20/train_v0.2.json"                                              # source 
data_t_path = "dataset/quac_context/QuAC20/val_v0.2.json"
data_new_path = save_path_pref + "train_v0.2.json"                                      # processed dataset 
data_new_t_path = save_path_pref + "val_v0.2.json"

# INIT
doc_list = doc_list_t = q_list = q_list_t = tfidf_list = tfidf_list_t = [] 

# LOAD DATA (cls/qas from ori, context from same cls)
print("== LOAD DATA ==")
with open(data_path, encoding = 'utf-8') as data_json, open(data_t_path, encoding = 'utf-8') as data_t_json:
    dataset = collections.OrderedDict(json.load(data_json))
    dataset_t = collections.OrderedDict(json.load(data_t_json))
new_data = dataset
new_data_t = dataset_t

for article in dataset['data']:
    cls_label = article['type_label']
    for paragraph in article['paragraphs']:
        # context = paragraph["context"]
        # if doc_list == []:
        #     doc_list = [context]
        # else:
            # doc_list.append(context)
        questions = []
        for qas in paragraph['qas']:
            questions.append(qas['question'])

        if q_list == []:
            q_list = [questions]
        else:
            q_list.append([questions])

    document = ""
    path = source_path_pref + 'cls_' + str(cls_label) + '.txt'
    # dataset_plsa/quac/complex/80/cls_80.txt
    with open(path, 'r') as file:           # file = codecs.open(path, 'r', 'utf-8')
        for doc in file:
            doc_line = doc.strip()
            document = document + ". " + doc_line
        # documents = [document.strip() for document in file] 
    file.close()

    if doc_list == []:
        doc_list = document
    else:
        doc_list.append(document)


for article in dataset_t['data']:
    cls_label = article['type_label']
    for paragraph in article['paragraphs']:
        # context = paragraph["context"]
        # if doc_list_t == []:
        #     doc_list_t = [context]
        # else:
        #     doc_list_t.append(context)
        questions = []
        for qas in paragraph['qas']:
            questions.append(qas['question'])

        if q_list_t == []:
            q_list_t = [questions]
        else:
            q_list_t.append([questions])

    document = ""   
    path = source_path_pref + 'cls_' + str(cls_label) + '.txt'
    with open(path, 'r') as file:           # file = codecs.open(path, 'r', 'utf-8')
        for doc in file:
            doc_line = doc.strip()
            document = document + ". " + doc_line
        # documents = [document.strip() for document in file] 
    file.close()

    if doc_list == []:
        doc_list = document
    else:
        doc_list.append(document)

# PICK IMPORTANT WORD (TFIDF)
tfidf = TfidfVectorizer()

for idx_doc, item_doc in enumerate(doc_list):
    for idx_q, item_q in enumerate(q_list):
        if idx_doc==idx_q:
            response = tfidf.fit_transform([item_doc, item_q])
            feature_names = tfidf.get_feature_names()
            for col in response.nonzero()[1]:
                print (feature_names[col], ' - ', response[0, col])
                # word: feature_names[col], score: response[0, col]

# extract TFIDF extend word
# check similarity with ori
# preprocess wd into New data


for article in dataset['data']:
    cls_label = article['type_label']
    for paragraph in article['paragraphs']:
        for qas in paragraph['qas']:
            # new qas [w0 w1 k0 k1 w2 ...]
            # kg label [0,0,1,1,0,...]
            continue
