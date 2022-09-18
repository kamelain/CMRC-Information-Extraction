
import json, numpy, collections

save_path_pref = 'dataset_wn/quac10_v5/'        # common prefix of processed documents  
data_path = "dataset/quac_bg/QuAC10/train_v0.2.json"                              # clustered source 
data_t_path = "dataset/quac_bg/QuAC10/val_v0.2.json"
# save_path_pref = 'dataset_wn/quac10_v3_small/'        # common prefix of processed documents  
# data_path = "dataset/quac_context/QuAC10/val_v0.2_small.json"                              # clustered source 
# data_t_path = "dataset/quac_context/QuAC10/val_v0.2_small.json"
data_new_path = save_path_pref + "train_v0.2.json"                                      # processed dataset 
data_new_t_path = save_path_pref + "val_v0.2.json"
q_wn_list_path = save_path_pref + "q_wn_list.json"

print(" === LOAD DATA === ")
with open(data_path, encoding = 'utf-8') as data_json, open(data_t_path, encoding = 'utf-8') as data_t_json:
    dataset = collections.OrderedDict(json.load(data_json))
    dataset_t = collections.OrderedDict(json.load(data_t_json))
new_data = dataset
new_data_t = dataset_t

# ===================== USING part-of-speech tagging ===================== 
import nltk
from nltk import FreqDist
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import wordnet, treebank
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util

# text = nltk.tokenize.word_tokenize("They refuse to permit us to obtain the refuse permit")
# nltk.pos_tag(text)
wnl = WordNetLemmatizer()
model = SentenceTransformer('all-MiniLM-L6-v2')
# NOUN, VERB, ADJECTIVE
tags = set(['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RP', 'JJ', 'JJR', 'JJS'])
tag_n = set(['NN', 'NNS', 'NNP', 'NNPS'])
tag_v = set(['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
tag_j = set(['JJ', 'JJR', 'JJS'])
tag_x = set(['VERB', 'NOUN'])

wsj = treebank.tagged_words(tagset='universal')
common_dict = nltk.FreqDist(wsj).most_common(150)
common_list = []
for ((wd,tag),cnt) in common_dict:
    if tag in tag_x:
        common_list.append(wd)
# common_list : ['is', 'said', '%', 'Mr.', 'was', 'are', 'be', 'has', 'have', 'will', 'company', 'U.S.', 'year', 'says', 'would', 'were', 'market', 'had', 'New', 'been', 'trading', 'stock', 'president', 'program', 'could', 'Corp.', 'share', 'years', 'shares', 'York', "'s", 'Inc.', 'can', 'prices', 'do', 'government', 'business', 'say', 'Japan', 'Co.', 'make', 'may', 'cents', 'funds', 'price', 'stocks', 'index', 'investors', 'companies', 'futures', 'time', 'did', 'rose', 'October', 'yesterday', 'sales']
common_list2 = ["", "Are", "Is", "Was", "Did", "did", "Do", "was",  "were", "be", "do", "have", "DO", "Be", "former", "come", "get", "represent", "interest", "coif", "look", "suffice", "occur", "WA", "exist", "record_album", "winnings", "playact", "set", "answer", "fare", "take", "Song", "yr", "have_got", "1st", "banding", "go", "constitute", "comprise", "interestingness", "do_work", "movie", "awarding", "follow", "Tell", "euphony", "embody", "early_on", "create", "matter", "track_record", "doh", "hitting", "noteworthy", "cause", "work_on"]
common_list = common_list + common_list2
q_wn_list = collections.OrderedDict()

# blank data
print(" === INIT BLANK Q_WN === ")
for article in new_data['data']:
    for paragraph in article['paragraphs']:
        for qas in paragraph['qas']:
            if 'question_wn' not in qas.keys():
                qas['question_wn'] = qas['question']
                qas['question_wn_list'] = []
for article in new_data_t['data']:
    for paragraph in article['paragraphs']:
        for qas in paragraph['qas']:
            if 'question_wn' not in qas.keys():
                qas['question_wn'] = qas['question']
                qas['question_wn_list'] = []

def not_overlap(w1:str, w2:str): 
    w_cnt = len(w1)-len(w2)     #w1大
    if w_cnt==1 or w_cnt==2:
        if w1 == w2+"s":
            return False
        elif w1 == w2+"es":
            return False
    w_cnt = len(w2)-len(w1)     #w1小
    if w_cnt==1 or w_cnt==2:
        if w2 == w1+"s":
            return False
        elif w2 == w1+"es":
            return False
    return True
            
print(" === INJECT WORDNET KNOWLEDGE (TRAINING) === ")
for article in new_data['data']:
    for paragraph in article['paragraphs']:
        for qas in paragraph['qas']:
            # item_q = "what happened in 1983?"
            item_q = qas['question']
            item_q_emb = model.encode(item_q)
            tokens_positions = list(WhitespaceTokenizer().span_tokenize(item_q))        # Tokenize to spans to get start/end positions: [(0, 3), (4, 9), ... ]
            tokens = WhitespaceTokenizer().tokenize(item_q)                             # Tokenize on a string lists: ["what", "happened", "in", ... ]
            tokens = nltk.pos_tag(tokens)                                               # Run Part-of-Speech tager
            words = []                                                                  # Iterate on each token
            words_syn = []
            for i in range(len(tokens)):
                text, tag = tokens[i]  # Get tag                                        # start, end = tokens_positions[i]  # Get token start/end
                if tag in tags:
                    # words.append((start, end, tag, text))                             # [(0, 4, 'WP', 'what'), (5, 13, 'VBD', 'happened'), (14, 16, 'IN', 'in'), (17, 22, 'CD', '1983?')]
                    words.append((tag, text))
            # print(words)
            q_ori_list = []
            kg_list = []
            q_kg = item_q
            for word in words:
                tag = word[0]
                wd = word[1]
                wd_ori = ""
                if tag in tag_n:
                    wd_ori = wnl.lemmatize(wd, 'n')
                    q_ori_list.append(wd_ori)
                elif tag in tag_v:
                    wd_ori = wnl.lemmatize(wd, 'v')
                    q_ori_list.append(wd_ori)
                elif tag in tag_j:
                    wd_ori = wnl.lemmatize(wd, 'a')
                    q_ori_list.append(wd_ori)
                else:
                    q_ori_list.append(wd_ori)

                wd_set = wordnet.synsets(wd_ori)
                # print(wd_set)
                best_score = 0
                best_wd = ""
                for s in wd_set:
                    s_list = s.lemma_names()        
                    for s_wd in s_list: 
                        # if wd_ori!=s_wd:
                        if wd_ori!=s_wd and \
                            s_wd not in common_list and \
                                not_overlap(wd,s_wd) and \
                                    s_wd not in q_ori_list:
                            q = item_q
                            q = q.replace(wd,s_wd)
                            q_emb = model.encode(q)
                            cos_sim = util.cos_sim(q_emb, item_q_emb)
                            if cos_sim>best_score:
                                best_score = cos_sim
                                best_wd = s_wd
                # print("best_score", best_score)
                # print("best_wd", best_wd)
                    
                # words_syn.append((wd, best_wd))
                # print(words_syn)
                kg_list.append(best_wd)

                if s_wd not in q_kg.split() and s_wd!="be" and s_wd!="do": 
                    q_kg = q_kg.replace(wd,wd + " " + s_wd)
                qas["question_wn"] = q_kg
                q_wn_list[qas["id"]] = q_kg
            qas["question_wn_list"] = kg_list
        # paragraph["kg_label"] = 1
                # ['happen', 'hap', 'go_on', 'pass_off', 'occur', 'pass', 'fall_out', 'come_about', 'take_place'] 
                # ['happen', 'befall', 'bechance'] 
                # ['happen', 'materialize', 'materialise'] 
                # ['find', 'happen', 'chance', 'bump', 'encounter']

print(" === INJECT WORDNET KNOWLEDGE (TESTING) === ")
for article in new_data_t['data']:
    for paragraph in article['paragraphs']:
        for qas in paragraph['qas']:
            item_q = qas['question']
            item_q_emb = model.encode(item_q)
            tokens_positions = list(WhitespaceTokenizer().span_tokenize(item_q))        # Tokenize to spans to get start/end positions: [(0, 3), (4, 9), ... ]
            tokens = WhitespaceTokenizer().tokenize(item_q)                             # Tokenize on a string lists: ["what", "happened", "in", ... ]
            tokens = nltk.pos_tag(tokens)                                               # Run Part-of-Speech tager
            words = []                                                                  # Iterate on each token
            words_syn = []
            for i in range(len(tokens)):
                text, tag = tokens[i]  # Get tag                                        # start, end = tokens_positions[i]  # Get token start/end
                if tag in tags:
                    words.append((tag, text))
            kg_list = []
            q_kg = item_q
            for word in words:
                tag = word[0]
                wd = word[1]
                wd_ori = ""
                if tag in tag_n:
                    wd_ori = wnl.lemmatize(wd, 'n')
                elif tag in tag_v:
                    wd_ori = wnl.lemmatize(wd, 'v')
                elif tag in tag_j:
                    wd_ori = wnl.lemmatize(wd, 'a')
                else:
                    pass

                wd_set = wordnet.synsets(wd_ori)
                best_score = 0
                best_wd = ""
                for s in wd_set:
                    s_list = s.lemma_names()        
                    for s_wd in s_list: 
                        # if wd_ori!=s_wd:
                        if wd_ori!=s_wd and s_wd not in common_list and not_overlap(wd,s_wd):
                            q = item_q
                            q = q.replace(wd,s_wd)
                            q_emb = model.encode(q)
                            cos_sim = util.cos_sim(q_emb, item_q_emb)
                            if cos_sim>best_score:
                                best_score = cos_sim
                                best_wd = s_wd
                kg_list.append(best_wd)

                if s_wd not in q_kg.split() and s_wd!="be" and s_wd!="do": 
                    q_kg = q_kg.replace(wd,wd + " " + s_wd)
                qas["question_wn"] = q_kg
                q_wn_list[qas["id"]] = q_kg
            qas["question_wn_list"] = kg_list
            

print(" === OUTPUT === ")
with open(data_new_path, "w") as n_data_file, \
    open(data_new_t_path, "w") as n_data_t_file, \
    open(q_wn_list_path, "w") as q_wn_list_file:
    json.dump(new_data, n_data_file)
    json.dump(new_data_t, n_data_t_file)
    json.dump(q_wn_list, q_wn_list_file)

# print(" === OUTPUT === ")
# with open(data_new_path, "w") as n_data_file, \
#     open(q_wn_list_path, "w") as q_wn_list_file:
#     json.dump(new_data, n_data_file)
#     json.dump(q_wn_list, q_wn_list_file)

#  ===================== USING TFIDF ===================== 

# TFIDF(doc, question) -> 重要字抓不出來

# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf = TfidfVectorizer()

# CASE 1 : TFIDF(doc, question), USING SIGNLE DOCUMENT
# item_doc = ""
# item_q = "what happened in 1983?"
# response = tfidf.fit_transform([item_doc, item_q])
# feature_names = tfidf.get_feature_names_out()
# for col in response.nonzero()[1]:
#     print(feature_names[col], ' - ', response[0, col])

# example
# 1983  -  0.0
# in  -  0.0
# happened  -  0.0
# what  -  0.0


# CASE 2 : TFIDF(docs, question), USING CLUSTERING DOCUMENTS
# num_clusters = 20                                                                       
# cls_type = 'context' 
# source_path_pref = 'dataset_plsa/quac/' + cls_type + '/' + str(num_clusters) + '/' 

# item_docs = [
#     "In May 1983, she married Nikos Karvelas, a composer, with whom she collaborated in 1975 and in November she gave birth to her daughter Sofia. After their marriage, she started a close collaboration with Karvelas. Since 1975, all her releases have become gold or platinum and have included songs by Karvelas. In 1986, she participated at the Cypriot National Final for Eurovision Song Contest with the song Thelo Na Gino Star (\"I Want To Be A Star\"), taking second place. This song is still unreleased up to date. In 1984, Vissi left her record company EMI Greece and signed with CBS Records Greece, which later became Sony Music Greece, a collaboration that lasted until 2013. In March 1984, she released Na 'Hes Kardia (\"If You Had a Heart\"). The album was certified gold. The following year her seventh album Kati Simveni (\"Something Is Happening\") was released which included one of her most famous songs, titled \"Dodeka\" [\"Twelve (O'Clock)\"] and reached gold status selling 80.000 units. In 1986 I Epomeni Kinisi (\"The Next Move\") was released. The album included the hit Pragmata (\"Things\") and went platinum, becoming the best selling record of the year. In February 1988 she released her ninth album Tora (\"Now\") and in December the album Empnefsi! (\"Inspiration!\") which went gold. In 1988, she made her debut as a radio producer on ANT1 Radio. Her radio program was titled after one of her songs Ta Koritsia Einai Atakta (\"Girls Are Naughty\") and was aired every weekend. In the same year, she participated with the song Klaio (\"I'm Crying\") at the Greek National Final for Eurovision Song Contest, finishing third. In 1989, she released the highly successful studio album Fotia (Fire), being one of the first albums to feature western sounds. The lead single Pseftika (\"Fake\") became a big hit and the album reached platinum status, selling 180.000 copies and becoming the second best selling record of 1990. She performed at \"Diogenis Palace\" in that same year, Athens's biggest nightclub/music hall at the time. CANNOTANSWER", \
#     "Although known primarily for R&B, g.o.d has often displayed their versatility as their albums feature songs which combine elements of different genres such as hip hop, rap, funk and dance pop: their debut single \"To Mother\" (eomeonimgge) contains elements of hip hop and a refrain sung in R&B style, the upbeat and rhythmic \"Friday Night\" from the second album heavily features funk elements and the ballad \"The Story of Our Lives\" from the most recent album utilizes a \"duet\" of the rap and singing parts in the chorus to create a polyphonic texture. Park Joon-hyung has stated that from the beginning he had intended for the group to effectively combine Western and Asian influences into their music. They were one of the few first generation K-pop groups to successfully incorporate elements of African American genres such as rap and hip hop into their music and utilize lines rapped entirely in Korean, which was uncommon at that time. While each song differs in genre, a key characteristic is a prominently distinct and simple melody which is accompanied by a minimalistic piano, guitar or percussion-based groove. Their overall style has been described as a mixture of \"belting styles\" with \"gentle rap, candid lyrics and plain vocal narration\". Their ability to seamlessly transition between rap and R&B is apparent in their hit songs, most of which are classified as R&B ballads or pop but contain at least several lines that are rapped. This combination of a prominent lyrical melody and rap has been described as their \"signature\". Some songs feature a spoken narration to introduce the song. They have shied away from trending genres such as bubblegum pop and electronic music. In contrast to their contemporaries such as Shinhwa, H.O.T. and Sechs Kies whose repertoire was largely either \"feel-good\" or hard rock or was choreography-based, g.o.d was focused on lyrical content, garnering them a wider demographic of fans. The trademark features of their repertoire are the \"story telling\" style and subject matter of their lyrics. MTV Asia describes their songs as \"[leaning] towards the social commentary side, with heartfelt lyrics that make people cry until today.\" Their songs often reflected relatable themes such as love, loss and family or were based on their personal experiences: their debut single \"To Mother\" was partly based on leader Park Joon-hyung's childhood while \"The Story of Five Men\" (daseos namja iyagi) described their frugal living conditions during their first year as struggling young singers. Other songs are more humorous and parodied the members themselves, such as the self-composed \"Sky Blue Balloon\" (haneulsaeg pungseon), in which rapper Danny Ahn refers to himself by his nickname Skinny Pig. Critics and media have noted the group's unique blend of five distinctly different voices. Mnet's Legends 100 series noted that \"[The] synergy created by these five people began a page of popular music history that has never been seen before.\" CANNOTANSWER", \
#     "Carter began his acting and singing career at a young age, when his mother enrolled him in voice lessons and dance lessons in ballet and tap at Karl and DiMarco's School of Theatre and Dance when he was 10. He performed in several commercials, such as the Florida State Lottery and The Money Store. He played the lead role in the fourth grade production of Phantom of the Opera at Miles Elementary School. He also did an educational video called \"Reach For The Book\", a show called \"The Klub\" and performing at the Tampa Bay Buccaneers home games for two years. He also made an appearance in the 1990 Johnny Depp film Edward Scissorhands as a child playing on a Slip 'N Slide. One of his dance teachers, Sandy, placed him in his first group called \"Nick and the Angels\". Between 1989-1993, Carter covered a various number of popular songs by other artists, including \"Breaking Up Is Hard to Do\" and \"Uptown Girl\" and a few original songs that he would perform at events. These recordings ended up on an unofficial release called Before the Backstreet Boys 1989-1993 by Dynamic Discs, Inc released in October 2002. It is revealed that through several auditions, Nick met AJ McLean and Howie Dorough and they became friends. After a troubled upbringing, he put in a winning performance on the 1992 New Original Amateur Hour at age 12. At 11 years old, Carter also auditioned for Disney's The Mickey Mouse Club and the Backstreet Boys around October 1992. He was not chosen right away to be a part of the Backstreet Boys because his mother Jane wanted him to join The Mickey Mouse Club so that he could stay in school. A week later, he was asked to join the group and was given the choice of either joining The Mickey Mouse Club with a $50,000 contract or this new music group. Nick chose to go with the group instead. The Mickey Mouse Club was cancelled a few years later. After he joined the group, he had his own personal tutor on tour. CANNOTANSWER", \
#     "In March 2011, Cruz premiered an all-new song called \"Telling the World\", which was written by Cruz and Alan Kasiyre for the Rio soundtrack. The track was not featured on any of Cruz's prior studio albums. On 23 May 2011, Cruz received his first Billboard Award in the United States. He announced onstage that his third album, entitled Black and Leather, would be released in the fourth quarter of 2011. In June 2011, Cruz released a collaborative single, \"Little Bad Girl\", with French DJ David Guetta and American rapper Ludacris. The song was a worldwide smash, topping singles charts across the world. During July and August 2011, a series of unreleased songs, recorded during the album sessions, were leaked to YouTube. in an interview, Cruz promised a \"fun\" and \"energetic\" album, and claimed that due to the leak, none of the tracks posted on YouTube would be included on the album. Cruz claimed, \"It's a shame really, because one person has spoilt it for all the fans. They could have had an album packed with 17, 18 tracks, and now, they're only getting eleven because of one stupid act of tomfooolery.\" During the interview, Cruz also claimed that the title of the album had been changed to Troublemaker, after one of the tracks included on the album. On 4 October 2011, the single \"Hangover\" was officially released in Germany and the US, and it was rumoured that a solo version of the song, that features on the physical release, would be included as a bonus track on the album. However, these rumours were later quashed. Around October 2011, the album was made available for pre-order, and as such, the album's final title, TY.O, was revealed. Cruz, via his Twitter account, made the decision to name the album TY.O after he expressed his annoyance at people constantly pronouncing his name wrong. The album was first released in Germany on 2 December 2011. It will be released in the UK on 31 December 2012. It was going to release in the United States on 17 May 2012, but it was released on 31 December 2012, coinciding with the UK release, but with new tracks, including the U.S. single \"Fast Car\". Cruz was scheduled to co-headline Pitbull's Australian leg of Planet Pit World Tour in August 2012. On 12 August, Cruz performed at the closing ceremony of the 2012 Summer Olympics in London. The performance at London's Olympic Stadium saw Cruz sing his own song \"Dynamite\", and also perform \"Written in the Stars\" alongside Jessie J and Tinie Tempah. CANNOTANSWER"        
# ]
# item_doc = ""
# item_q = "what happened in 1983?"

# for item in item_docs:
#     item_doc = item_doc + ". " + item

# print("Document: ", item_doc)
# print("Question: ", item_q)

# response = tfidf.fit_transform([item_doc, item_q])
# feature_names = tfidf.get_feature_names()
# for col in response.nonzero()[1]:
#     q_words = item_q.split()
#     for word in q_words:
#         if feature_names[col]==word:
#             print(feature_names[col], ' - ', response[0, col])
#     # word: feature_names[col], score: response[0, col]

# example
# in  -  0.19886473111986389
# happened  -  0.0
# what  -  0.0
# in  -  0.19886473111986389

