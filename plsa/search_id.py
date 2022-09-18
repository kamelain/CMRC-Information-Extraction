import json, collections

id_file_path_prefix = 'dataset_context_line/train_id/'

data_path = "dataset/quac_context/QuAC10/train_v0.2.json"                                # clustered source 
data_new_path = "dataset_plsa/tag/train_v0.2_context.json"                                # clustered source 

with open(data_path, encoding = 'utf-8') as data_json:
    dataset = collections.OrderedDict(json.load(data_json))
new_data = dataset

print(" === TAGGING === ")
for article in new_data['data']:
    for paragraph in article['paragraphs']:
        id_a = paragraph['id']
        
        for i in range(1, 12):
            id_file_path = id_file_path_prefix + '_id_list_' + str(i)
            
            with open(id_file_path, encoding = 'utf-8') as train_id_file:
                id_set = list(json.load(train_id_file))

            for id_b in id_set:
                if id_a==id_b:
                    paragraph['tag'] = i


print(" === OUTPUT === ")
with open(data_new_path, "w") as n_data_file:
    json.dump(new_data, n_data_file)
