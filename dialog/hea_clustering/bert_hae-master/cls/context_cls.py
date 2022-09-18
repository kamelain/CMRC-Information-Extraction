import json, numpy, collections, sys
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# https://www.sbert.net/docs/quickstart.html
# https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/clustering/kmeans.py
# CUDA_VISIBLE_DEVICES=0 python context_cls.py

num_clusters = int(sys.argv[1])
clu_type = sys.argv[2]

# num_clusters = 40       # for KMAENS
# num_clusters = 110  #510: 0.5/10  for DBSCAN
# dbscan_eps = 1
# dbscan_cnt = 20
train_len = test_len = 0       #train: 11567, test: 1000
data_list = data_list_t = data_emb = new_list = label_list = []     #context
cnt_list = collections.OrderedDict()

data_path = "../QuAC/train_v0.2.json"
data_t_path = "../QuAC/val_v0.2.json"
data_new_path = sys.argv[3]
data_new_t_path = sys.argv[4]
emb_path = sys.argv[5]
cnt_path = sys.argv[6]
fig_path = sys.argv[7]
fig_path2 = sys.argv[8]

with open(data_path, encoding = 'utf-8') as data_json, open(data_t_path, encoding = 'utf-8') as data_t_json:
    dataset = collections.OrderedDict(json.load(data_json))
    dataset_t = collections.OrderedDict(json.load(data_t_json))
new_data = dataset
new_data_t = dataset_t

print("== LOAD DATA ==")
if str(sys.argv[2]) == "context":
# context as embedding
    for article in dataset['data']:
        for paragraph in article['paragraphs']:
            if data_list == []:
                data_list = [paragraph["context"]]
            else:
                data_list.append(paragraph["context"])
    for article in dataset_t['data']:
        for paragraph in article['paragraphs']:
            if data_list_t == []:
                data_list_t = [paragraph["context"]]                
            else:
                data_list_t.append(paragraph["context"])

elif str(sys.argv[2]) == "bg":
# background as embedding
    for article in dataset['data']:
        if data_list == []:
            data_list = [article["background"]]
        else:
            data_list.append(article["background"])
    for article in dataset_t['data']:
        if data_list_t == []:
            data_list_t = [article["background"]]                
        else:
            data_list_t.append(article["background"])

elif str(sys.argv[2]) == "title":
# title/section title as embedding
    for article in dataset['data']:
        title = [article["title"]]
        s_title = [article["section_title"]]
        new_title = title + s_title
        if data_list == []:
            data_list = [new_title]
        else:
            data_list.append(new_title)
    for article in dataset_t['data']:
        if data_list_t == []:
            data_list_t = [new_title]                
        else:
            data_list_t.append(new_title)

train_len = len(data_list)
test_len = len(data_list_t)
data_list = data_list + data_list_t
print("train_len", train_len)
print("test_len", test_len)

# Embedding
print("== BUILD MODEL ==")
model = SentenceTransformer('all-mpnet-base-v2')
print("== ENCODE DATA ==")
data_emb = model.encode(data_list)
print("== DATA TRANSFORM ==")
data_emb = numpy.array(data_emb)

# cluster
print("== CLUSTERING ==")
cluster = KMeans(n_clusters=num_clusters).fit(data_emb)
# cluster = DBSCAN(eps=dbscan_eps,min_samples=dbscan_cnt).fit(data_emb)
label_list = list(cluster.labels_)
print("label_list", label_list)

# scatter plot
print("== PLOT ==")
L_sk = PCA(2).fit_transform(data_emb)
plt.scatter(L_sk[:,0],L_sk[:,1],c=cluster.labels_, cmap='Spectral')  
plt.savefig(fig_path)

# output 
print("== LABELING ==")
for i in label_list:
    i = int(i)
    new_list.append(i)
for i in new_list:
    if i not in cnt_list:
        cnt_list[i] = 1
    else:
        cnt_list[i] = cnt_list[i] + 1

plt.hist(label_list, color='#34B3F1')
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.savefig(fig_path2)

print("label list: ", len(label_list))
for idx, val in enumerate(label_list):
    if idx < train_len:
        new_data['data'][idx]['type_label'] = int(val)
    else:
        idx = idx-train_len
        new_data_t['data'][idx]['type_label'] = int(val)

print("== OUTPUTING ==")
# with open(emb_path, "w") as emb_file, \
#     open(cnt_path, "w") as cnt_file, \
#     open(data_new_path, "w") as n_data_file, \
#     open(data_new_t_path, "w") as n_data_t_file:
#     json.dump(new_list, emb_file)
#     json.dump(cnt_list, cnt_file)
#     json.dump(new_data, n_data_file)
#     json.dump(new_data_t, n_data_t_file)
with open(cnt_path, "w") as cnt_file:
    json.dump(cnt_list, cnt_file)
    