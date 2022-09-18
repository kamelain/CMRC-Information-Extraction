import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import WhitespaceTokenizer
from nltk import FreqDist
# from nltk.stem import WordNetLemmatizer

# nltk.data.path.append("/share/nas167/chinyingwu/nlp/wordnet")
# nltk.download('wordnet')
# nltk.download()

tags = set(['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RP', 'JJ', 'JJR', 'JJS'])
tag_n = set(['NN', 'NNS', 'NNP', 'NNPS'])
tag_v = set(['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
tag_j = set(['JJ', 'JJR', 'JJS'])
tag_x = set(['VERB', 'NOUN'])
# wnl = WordNetLemmatizer()

item_q = "did they have any children?"
tokens_positions = list(WhitespaceTokenizer().span_tokenize(item_q))        # Tokenize to spans to get start/end positions: [(0, 3), (4, 9), ... ]
tokens = WhitespaceTokenizer().tokenize(item_q)                             # Tokenize on a string lists: ["what", "happened", "in", ... ]
tokens = nltk.pos_tag(tokens) 
words = []      

wsj = nltk.corpus.treebank.tagged_words(tagset='universal')
# word_tag_fd = nltk.FreqDist(wsj)
common_dict = nltk.FreqDist(wsj).most_common(150)
common_list = []
for ((wd,tag),cnt) in common_dict:
    # print("wd: ", wd, "tag:", tag)
    if tag in tag_x:
        common_list.append(wd)
print("common_list", common_list)
# print("common_dict: ", common_dict)

# for (wt, _) in word_tag_fd.most_common(10):
#     if wt[1] == 'VERB':
        
# common_dict = [wt[0] for (wt, _) in word_tag_fd.most_common() if wt[1] == 'VERB']
# print("common_dict: ", common_dict)

# for i in range(len(tokens)):
#     text, tag = tokens[i]  # Get tag                                        # start, end = tokens_positions[i]  # Get token start/end
#     if tag in tags:
#         # words.append((start, end, tag, text))                             # [(0, 4, 'WP', 'what'), (5, 13, 'VBD', 'happened'), (14, 16, 'IN', 'in'), (17, 22, 'CD', '1983?')]
#         words.append((tag, text))
# print(words)




# # 單義詞
# print(wn.synsets('motorcar'))           
#     # printout : [Synset('car.n.01')]
# print(wn.synsets('motorcar')[0].lemma_names()) 
#     # ['car', 'auto', 'automobile', 'machine', 'motorcar']

# # 多義詞
# print(wn.synsets('trunk'))
#     # printout : [Synset('trunk.n.01'), Synset('trunk.n.02'), Synset('torso.n.01'), Synset('luggage_compartment.n.01'), Synset('proboscis.n.02')]
# print(wn.synsets('trunk')[0].lemma_names())
# print(wn.synsets('trunk')[1].lemma_names())
#     # ['trunk', 'tree_trunk', 'bole']
#     # ['trunk']

# # 找兩個詞 truck 和 car 之間的最低共同詞
# t = wn.synsets('truck')[0]
# c = wn.synsets('car')[0]
# print(t.lowest_common_hypernyms(c))
# print(t.lowest_common_hypernyms(c)[0].lemma_names())
#     # truck, motor
#     #     [Synset('instrumentality.n.03')]
#     #     ['instrumentality', 'instrumentation']

#     # truck, car
#     #     [Synset('motor_vehicle.n.01')]
#     #     ['motor_vehicle', 'automotive_vehicle']

# # 尋找 motorcar 的上位詞組
# m = wn.synsets('car')[0]
# print(m.hypernyms())
# print(m.hypernyms()[0].lemma_names())
#     # [Synset('motor_vehicle.n.01')]