import json, os
from plsa import run_plsa, preprocessing

# train_data_path = "dataset_output_small/"
# train_id_path = "dataset_output_small/_id_list.txt"

# train_data_path = "dataset_context_line/test/"
# train_id_path = "dataset_context_line/test/_id_list.txt"

train_data_path = "dataset_context_line/train/"
# train_save_path = "dataset_context_line/train_save/"
train_id_path = "dataset_context_line/train_id/"

stopwords_file_path = "stopwords.dic"
K = 10                  # K, number of topic
max_iteration = 30
threshold = 10.0
topic_words_num = 2

# run_plsa(
#   datasetFilePath:str, stopwordsFilePath:str, K:int, 
#   maxIteration:int, threshold:float, topicWordsNum:int, 
#   docTopicDist:str, topicWordDist:str, dictionary:str, topicWords:str):

#1-5, 5-9, 9-12

for i in range(11,12):
    train_id_path = "dataset_context_line/train_id/" + '_id_list_' + str(i) 
    train_save_path = "dataset_context_line/train_save/" + str(i) + '/'

    with open(train_id_path, encoding = 'utf-8') as train_id_file:
        id_set = list(json.load(train_id_file))

    for id in id_set:
        postfix = id + "(2)"
        dataset_file_path = train_data_path + id + '.txt'

        doc_topic_dist = train_save_path + postfix + '_doc_topic_distribution.txt'
        topic_word_dist = train_save_path + postfix + '_topic_word_distribution.txt'
        dictionary = train_save_path + postfix + '_dictionary.dic'
        topic_words = train_save_path + postfix + '_topics.txt'

        N, M, word2id, id2word, X = preprocessing(dataset_file_path, stopwords_file_path)
        run_plsa(id2word, X, N, M, K, max_iteration, threshold, topic_words_num, \
                doc_topic_dist, topic_word_dist, dictionary, topic_words)


# with open(train_id_path, encoding = 'utf-8') as train_id_file:
#     id_set = list(json.load(train_id_file))

# for id in id_set:
#     dataset_file_path = train_save_path + id + '.txt'

#     postfix = id + "(2)"
#     os.mkdir(train_data_path+postfix)

#     doc_topic_dist = train_data_path + postfix + '_doc_topic_distribution.txt'
#     topic_word_dist = train_data_path + postfix + '_topic_word_distribution.txt'
#     dictionary = train_data_path + postfix + '_dictionary.dic'
#     topic_words = train_data_path + postfix + '_topics.txt'

#     N, M, word2id, id2word, X = preprocessing(dataset_file_path, stopwords_file_path)
#     run_plsa(id2word, X, N, M, K, max_iteration, threshold, topic_words_num, \
#             doc_topic_dist, topic_word_dist, dictionary, topic_words)