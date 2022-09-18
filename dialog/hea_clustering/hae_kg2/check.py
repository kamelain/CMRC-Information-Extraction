###### CHECK KG INPUT 

q = "what happened in 1983?"
q_wn = "what happened occur in 1983?"
q_wn_list = ["occur"]

# q = "what happened in 1983?"
# q_wn = "did they have any children?"
# q_wn_list = ["occur"]

question_text = q_wn

question_new_text = ""
kg_idx = []
kg_idx2 = []
kg_list = q_wn_list       # qa["question_wn_list"]
q_s = question_text.split()
for idx_q, wd_q in enumerate(q_s):
    for idx_kg, wd_kg in enumerate(kg_list):
        if wd_q==wd_kg:
            real_idx = idx_q+1              # +1 for [CLS] index
            print("real_idx: ", real_idx)               # real_idx: 3, kg_idx: [3]
            
            # kg_idx.append(real_idx)                   # for + K-BERT (no relation)    + entity

            # Synonym (for K-BERT)
            kg_idx_offset = len(kg_idx)//2   # count "Synonym" offset
            kg_idx2.append(real_idx+kg_idx_offset)       # for + K-BERT (with relation), + entity
            kg_idx2.append(real_idx+kg_idx_offset+1)     #                               + relation
            question_new_text = question_new_text + " synonym"
    question_new_text = question_new_text + " " + wd_q
question_text = question_new_text                       

# print("kg_idx: ", kg_idx)
print("question_text: ", question_text)
print("kg_idx2: ", kg_idx2)



###### CHECK POSITION EMBEDDING

data = {"data": [{"paragraphs": [{"context": "In September 2016 Vladimir Markin, official spokesman for the Investigative Committee, included the killing of Anna Politkovskaya among the Most Dramatic Crimes in 21st century Russia and claimed that it had been solved. Her colleagues at Novaya gazeta protested that until the instigator or sponsor of the crime was identified, arrested and prosecuted the case was not closed. On 7 October 2016 Novaya gazeta released a video clip of its editors, correspondents, photographers and technical and administrative staff holding text-boards giving details of the case and stating, repeatedly, \"The sponsor of Anna's murder has not been found\". On the same day deputy chief editor Sergei Sokolov published a damning summary of the official investigation, describing its false turns and shortcomings, and emphasised that it had now effectively been wound up. After the three Makhmudov brothers, Khadjikurbanov and Lom-Ali Gaitukayev were convicted in 2014, wrote Sokolov, the once large team of investigators was reduced to one person and within a year he retired, to be replaced by a lower-ranking investigator. In accordance with Russian law there is a 15-year statute of limitation for the \"particularly grave\" crime of first degree murder. The 2000 killing of Igor Domnikov, another Novaya gazeta journalist, showed that the perpetrators might be identified (they were convicted in 2008), as was the businessman-intermediary who hired them (he was sentenced in December 2013 to seven years' imprisonment). The man allegedly responsible for ordering the attack on Domnikov was brought to court in 2015. In May that year the case against him was discontinued because the statute of limitations had expired. The Intercept published a top-secret document released by Edward Snowden with a screenshot of Intellipedia according to which (TS//SI/REL TO USA, AUS, CAN, GBR, NZL) Russian Federal Intelligence Services (probably FSB) are known to have targeted the webmail account of the murdered Russian journalist Anna Politkovskaya. On 5 December 2005, RFIS initiated an attack against the account annapolitovskaya@US Provider1, by deploying malicious software which is not available in the public domain. It is not known whether the attack is in any way associated with the death of the journalist.[1] CANNOTANSWER", "qas": [{"followup": "n", "yesno": "y", "question": "Did they have any clues?", "answers": [{"text": "Her colleagues at Novaya gazeta protested that until the instigator or sponsor of the crime was identified, arrested and prosecuted the case was not closed.", "answer_start": 221}, {"text": "Novaya gazeta released a video clip of its editors, correspondents, photographers and technical and administrative staff holding text-boards giving details of the case", "answer_start": 396}, {"text": "probably FSB) are known to have targeted the webmail account of the murdered Russian journalist Anna Politkovskaya.", "answer_start": 1908}], "id": "C_0aaa843df0bd467b96e5a496fc0b033d_1_q#0", "orig_answer": {"text": "probably FSB) are known to have targeted the webmail account of the murdered Russian journalist Anna Politkovskaya.", "answer_start": 1908}}, {"followup": "n", "yesno": "x", "question": "How did they target her email?", "answers": [{"text": "RFIS initiated an attack against the account annapolitovskaya@US Provider1, by deploying malicious software which is not available in the public domain.", "answer_start": 2044}, {"text": "On 5 December 2005, RFIS initiated an attack against the account annapolitovskaya@US Provider1, by deploying malicious software", "answer_start": 2024}, {"text": "On 5 December 2005, RFIS initiated an attack against the account annapolitovskaya@US Provider1, by deploying malicious software which is not available in the public domain.", "answer_start": 2024}, {"text": "RFIS initiated an attack against the account annapolitovskaya@US Provider1, by deploying malicious software which is not available in the public domain.", "answer_start": 2044}, {"text": "On 5 December 2005, RFIS initiated an attack against the account annapolitovskaya@US Provider1, by deploying malicious software", "answer_start": 2024}], "id": "C_0aaa843df0bd467b96e5a496fc0b033d_1_q#1", "orig_answer": {"text": "On 5 December 2005, RFIS initiated an attack against the account annapolitovskaya@US Provider1, by deploying malicious software", "answer_start": 2024}}, {"followup": "n", "yesno": "x", "question": "Did they get into trouble for that?", "answers": [{"text": "CANNOTANSWER", "answer_start": 2294}, {"text": "CANNOTANSWER", "answer_start": 2294}, {"text": "CANNOTANSWER", "answer_start": 2294}], "id": "C_0aaa843df0bd467b96e5a496fc0b033d_1_q#2", "orig_answer": {"text": "CANNOTANSWER", "answer_start": 2294}}, {"followup": "n", "yesno": "y", "question": "Did they have any murder suspects?", "answers": [{"text": "Khadjikurbanov and Lom-Ali Gaitukayev were convicted in 2014,", "answer_start": 889}, {"text": "the three Makhmudov brothers, Khadjikurbanov and Lom-Ali Gaitukayev were convicted in 2014,", "answer_start": 859}, {"text": "After the three Makhmudov brothers, Khadjikurbanov and Lom-Ali Gaitukayev were convicted in 2014,", "answer_start": 853}, {"text": "After the three Makhmudov brothers, Khadjikurbanov and Lom-Ali Gaitukayev were convicted in 2014,", "answer_start": 853}], "id": "C_0aaa843df0bd467b96e5a496fc0b033d_1_q#3", "orig_answer": {"text": "After the three Makhmudov brothers, Khadjikurbanov and Lom-Ali Gaitukayev were convicted in 2014,", "answer_start": 853}}, {"followup": "m", "yesno": "x", "question": "Did they go to jail?", "answers": [{"text": "CANNOTANSWER", "answer_start": 2294}, {"text": "CANNOTANSWER", "answer_start": 2294}, {"text": "CANNOTANSWER", "answer_start": 2294}], "id": "C_0aaa843df0bd467b96e5a496fc0b033d_1_q#4", "orig_answer": {"text": "CANNOTANSWER", "answer_start": 2294}}, {"followup": "y", "yesno": "x", "question": "Is there anything else interesting in the article?", "answers": [{"text": "the killing of Anna Politkovskaya among the Most Dramatic Crimes in 21st century Russia and claimed that it had been solved.", "answer_start": 96}, {"text": "It is not known whether the attack is in any way associated with the death of the journalist.[1]", "answer_start": 2197}, {"text": "Her colleagues at Novaya gazeta protested that until the instigator or sponsor of the crime was identified, arrested and prosecuted the case was not closed.", "answer_start": 221}, {"text": "On 7 October 2016 Novaya gazeta released a video clip of its editors, correspondents, photographers and technical and administrative staff holding text-boards giving details of the case", "answer_start": 378}, {"text": "In accordance with Russian law there is a 15-year statute of limitation for the \"particularly grave\" crime of first degree murder.", "answer_start": 1107}], "id": "C_0aaa843df0bd467b96e5a496fc0b033d_1_q#5", "orig_answer": {"text": "In accordance with Russian law there is a 15-year statute of limitation for the \"particularly grave\" crime of first degree murder.", "answer_start": 1107}}, {"followup": "n", "yesno": "n", "question": "Are they close to solving it?", "answers": [{"text": "(they were convicted in 2008), as was the businessman-intermediary who hired them (he was sentenced in December 2013 to seven years' imprisonment).", "answer_start": 1356}, {"text": "the once large team of investigators was reduced to one person and within a year he retired, to be replaced by a lower-ranking investigator.", "answer_start": 966}, {"text": "In May that year the case against him was discontinued because the statute of limitations had expired.", "answer_start": 1600}], "id": "C_0aaa843df0bd467b96e5a496fc0b033d_1_q#6", "orig_answer": {"text": "In May that year the case against him was discontinued because the statute of limitations had expired.", "answer_start": 1600}}, {"followup": "n", "yesno": "x", "question": "Is it similar to any other cases?", "answers": [{"text": "). The man allegedly responsible for ordering the attack on Domnikov was brought to court in 2015.", "answer_start": 1501}, {"text": "The 2000 killing of Igor Domnikov, another Novaya gazeta journalist, showed that the perpetrators might be identified", "answer_start": 1238}, {"text": "The Intercept published a top-secret document released by Edward Snowden with a screenshot of Intellipedia according to which", "answer_start": 1703}], "id": "C_0aaa843df0bd467b96e5a496fc0b033d_1_q#7", "orig_answer": {"text": "The Intercept published a top-secret document released by Edward Snowden with a screenshot of Intellipedia according to which", "answer_start": 1703}}], "id": "C_0aaa843df0bd467b96e5a496fc0b033d_1"}], "section_title": "The murder remains unsolved, 2016", "background": "Anna Stepanovna Politkovskaya (Russian: Anna Stepanovna Politkovskaia, IPA: ['an:@ sjtjI'pan@vn@ p@ljIt'kofsk@j@]; Ukrainian: Ganna Stepanivna Politkovs'ka ['han:a ste'panjiuna poljit'kousjka]; nee Mazepa [ma'zepa]; 30 August 1958 - 7 October 2006) was a Russian journalist, writer, and human rights activist who reported on political events in Russia, in particular, the Second Chechen War (1999-2005).", "title": "Anna Politkovskaya", "type_label": 1}]}

doc = "In September 2016 Vladimir Markin, official spokesman for the Investigative Committee, included the killing of Anna Politkovskaya among the Most Dramatic Crimes in 21st century Russia and claimed that it had been solved. Her colleagues at Novaya gazeta protested that until the instigator or sponsor of the crime was identified, arrested and prosecuted the case was not closed. On 7 October 2016 Novaya gazeta released a video clip of its editors, correspondents, photographers and technical and administrative staff holding text-boards giving details of the case and stating, repeatedly, \"The sponsor of Anna's murder has not been found\". On the same day deputy chief editor Sergei Sokolov published a damning summary of the official investigation, describing its false turns and shortcomings, and emphasised that it had now effectively been wound up. After the three Makhmudov brothers, Khadjikurbanov and Lom-Ali Gaitukayev were convicted in 2014, wrote Sokolov, the once large team of investigators was reduced to one person and within a year he retired, to be replaced by a lower-ranking investigator. In accordance with Russian law there is a 15-year statute of limitation for the \"particularly grave\" crime of first degree murder. The 2000 killing of Igor Domnikov, another Novaya gazeta journalist, showed that the perpetrators might be identified (they were convicted in 2008), as was the businessman-intermediary who hired them (he was sentenced in December 2013 to seven years' imprisonment). The man allegedly responsible for ordering the attack on Domnikov was brought to court in 2015. In May that year the case against him was discontinued because the statute of limitations had expired. The Intercept published a top-secret document released by Edward Snowden with a screenshot of Intellipedia according to which (TS//SI/REL TO USA, AUS, CAN, GBR, NZL) Russian Federal Intelligence Services (probably FSB) are known to have targeted the webmail account of the murdered Russian journalist Anna Politkovskaya. On 5 December 2005, RFIS initiated an attack against the account annapolitovskaya@US Provider1, by deploying malicious software which is not available in the public domain. It is not known whether the attack is in any way associated with the death of the journalist."
q_wn = "what happened occur in 1983?"
#   [CLS]', 'what', 'happened', 'occur', 'in', '1983?', '[SEP]
#    0       1       2          3       3       4       5
q_wn_list = ["occur"]
q_idx_list = [3]

q_wn_k = "what happened synonym occur in 1983?"
q_wn_list = ["occur"]
q_idx_list_k = [3, 4]
#   [CLS]', 'what', 'happened', 'synonym', 'occur', 'in', '1983?', '[SEP]
#    0       1       2          3           4       3       4       5
#   [3,4]

tokens = []
segment_ids = []
cls_ids = []
kpos_ids = []
kpos = 0

# [CLS]
tokens.append("[CLS]")
cls_ids.append(0)
kpos_ids.append(kpos)

# # Question   ##########   (Kpos)
for q_wd in q_wn.split():
    tokens.append(q_wd)
    cls_ids.append(0)
    situate = 'a'

    if kpos not in q_idx_list:
        kpos = kpos + 1
        situate = 'b'
    elif kpos_ids.count(kpos)==2:
        kpos = kpos + 1
        situate = 'c'
        
    print(q_wd, " ,", kpos, " ,", situate, kpos_ids)
    kpos_ids.append(kpos)
    # print("token: ", q_wd, " pos: ", kpos )
    
#     # for idx in q_idx_list:           # K-BERT (no synonyms)     # append的pos為kg, pos扣回1
#     #     if kpos == idx:
#     #         kpos = kpos-1

# Question   ##########   (Kbert)
kg_done_list = []

for q_wd in q_wn_k.split():
    tokens.append(q_wd)
    cls_ids.append(0)
    situate = 'a, -'

    kpos = kpos + 1 
    kpos_ids.append(kpos)

    if kpos in q_idx_list_k: 
        if q_idx_list_k.index(kpos)%2!=0 and kpos not in kg_done_list: 
            kg_done_list.append(kpos)
            kpos = kpos - 2
            situate = 'b, syn+ent end'

    # if kpos not in q_idx_list_k:
    #     kpos = kpos + 1
    #     situate = 'b, not keyword'
    # elif kpos%2==0:
    #     kpos = kpos + 1
    #     situate = 'd, keyword'
    # elif kpos%2!=0 and kpos_ids.count(kpos)==2:
    #     kpos = kpos + 1
    #     situate = 'c, syn'
        
    print(q_wd, " ,", kpos, " ,", situate, kpos_ids)
    # kpos_ids.append(kpos)

    # if kpos%2!=0 and kpos_ids.count(kpos)==2:
    #     kpos = kpos - 2

    # print("token: ", q_wd, " pos: ", kpos )
    
    # for idx in q_idx_list:           # K-BERT (no synonyms)     # append的pos為kg, pos扣回1
    #     if kpos == idx:
    #         kpos = kpos-1

# [SEP]
tokens.append("[SEP]")
cls_ids.append(0)
kpos = kpos + 1
kpos_ids.append(kpos)

# # Document
# for d_wd in doc.split():
#     tokens.append(d_wd)
#     segment_ids.append(1)
#     cls_ids.append(1)
#     kpos = kpos + 1
#     kpos_ids.append(kpos)

# # [SEP]
# tokens.append("[SEP]")
# segment_ids.append(1)
# cls_ids.append(1)
# kpos = kpos + 1
# kpos_ids.append(kpos)

print("\n=====\n")
print("tokens", tokens)
print("cls_ids", cls_ids)
print("kpos_ids", kpos_ids)
print("kpos_ans", [0, 1, 2, 3, 4, 3, 4, 5])