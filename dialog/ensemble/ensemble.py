# -*- coding: utf8 -*-
import os
import json
import collections

ensemble_name = "excordc_v10"
# exc_path = "excord/output/result/nbest_predictions_.json"
# exc_path = "excord-ori/pretrain_model/best_model/nbest_predictions_.json"
exc_path = "/share/nas167/chinyingwu/nlp/dialog/excord/clus2/output/_context/nbest_predictions_.json"
hae_path = "/share/nas167/chinyingwu/nlp/dialog/excord/clus2/output/v10/nbest_predictions_.json"
# exc_path = "haec/bert_out/bg_40/nbest_predictions_24000.json"
# hae_path = "haec/bert_out/bg_10/nbest_predictions_24000.json"
hae_pre_path = "haec/bert_out/bg_10/predictions_24000.json"
n_best_path = "output/" + ensemble_name + "/nbest_predictions.json"
predict_path = "output/" + ensemble_name + "/predictions.json"
predict_ori_path = "output/" + ensemble_name + "/predictions_ori.json"

predict = collections.OrderedDict()
n_best = collections.OrderedDict()
predict_ori = collections.OrderedDict()

a = 2   # model weight 1 (better)
b = 1   # model weight 2
c = 0.81 # once panelty, 1 == no panelty

# load prediction files
with open(exc_path, encoding = 'utf-8') as exc_json:
    exc = collections.OrderedDict(json.load(exc_json))
with open(hae_path, encoding = 'utf-8') as hae_json:
    hae = collections.OrderedDict(json.load(hae_json))

# average
for key_exc, value_exc in exc.items():
    for key_hae, value_hae in hae.items():
        if key_exc == key_hae:
            prob = 0
            for ans_exc in value_exc:
                for ans_hae in value_hae:
                    if ans_exc["text"] == ans_hae["text"]:
                        temp = {}     
                        text = ans_hae["text"]
                        temp["text"] = text
                        # temp["probability"] = (ans_exc["probability"]+ans_hae["probability"])/2
                        temp["probability"] = (ans_exc["probability"]*a+ans_hae["probability"]*b)/(a+b)
                        if key_exc not in n_best:
                            n_best[key_exc] = []
                            n_best[key_exc].append(temp)   
                        else:
                            n_best[key_exc].append(temp)  

                        if key_exc not in predict:
                            predict[key_exc] = {}
                            predict[key_exc] = temp["text"]
                            # prob = temp["probability"]
                            prob = temp["probability"]*c
                        elif temp["probability"] > prob:
                            predict[key_exc] = temp["text"]
                            # prob = temp["probability"]
                            prob = temp["probability"]*c

                if key_exc not in predict:
                    max_exc = max(value_exc, key=lambda x:x['probability'])
                    max_hae = max(value_hae, key=lambda x:x['probability'])
                    
                    # if max_exc['probability'] > max_hae['probability']*0.67:
                    if max_exc['probability'] > max_hae['probability']:
                        predict[key_exc] = max_exc['text']
                    else:
                        predict[key_exc] = max_hae['text']

# compare best with ori_predict, save best as ori format
for line in open(hae_pre_path, 'r'):
    if line.strip():
        pred_data = json.loads(line.strip())
        for idx, dia_id in enumerate(pred_data['qid']):
            for best_id, best_text in predict.items(): 
                if dia_id == best_id: 
                    pred_data['best_span_str'][idx] = best_text
        
    with open(predict_ori_path, 'a') as predict_ori:
        json.dump(pred_data, predict_ori)
        predict_ori.write('\n') 

# for best_id, best_text in predict.items():
#     for line in open(hae_pre_path, 'r'):
#         if pred_data is None:
#             if line.strip():
#                 pred_data = json.loads(line.strip())
#             # temp_doc = pred_data
#         for idx, dia_id in enumerate(pred_data['qid']):
#             if best_id == dia_id:
#                 qid = best_id.split("_q#")[0]
#                 pred_data['best_span_str'][idx] = best_text
                # predict_ori[best_id] = pred_data
            # with open(predict_ori_path, 'a') as predict_ori:
            #     json.dump(pred_data, predict_ori)
            #     predict_ori.write('\n')

# for best_id, best_text in predict.items():
#     done_list = {}
#     qid = best_id.split("_q#")[0]
#     for line in open(hae_pre_path, 'r'):
#         if line.strip():
#             pred_data = json.loads(line.strip())
#             print("pred_data['qid'] size", len(pred_data['qid']))
#             for idx, dia_id in enumerate(pred_data['qid']):
#                 if best_id == dia_id:
#                     pred_data['best_span_str'][idx] = best_text
#                     if qid not in done_list.keys():
#                         done_list[qid] = 1
#                     else:
#                         done_list[qid] = done_list[qid] + 1
#                     print("qid, size", qid, done_list[qid])
                    
#                     if done_list[qid] == len(pred_data['qid']):
#                         with open(predict_ori_path, 'a') as predict_ori:
#                             json.dump(pred_data, predict_ori)
#                             predict_ori.write('\n')
                    

    # temp_doc["best_span_str"][idx] = text_best
    
    # dia_id2 = pred_idx['qid'][0].split("_q#")[0]
    
    # for idx, id_key in enumerate(temp_doc['qid']):
    #     if id_key == key_exc:
    #         temp_doc["best_span_str"][idx] = text_best
    #         with open(predict_ori_path, 'a') as predict_ori:
    #             json.dump(temp_doc, predict_ori)
    #             predict_ori.write('\n')
        


# for line in open(hae_pre_path, 'r'):
#     if line.strip():
#         pred_idx = json.loads(line.strip())

#         temp_doc["best_span_str"][idx] = text_best
        
#         dia_id2 = pred_idx['qid'][0].split("_q#")[0]
        
#         for idx, id_key in enumerate(temp_doc['qid']):
#             if id_key == key_exc:
#                 temp_doc["best_span_str"][idx] = text_best
#                 with open(predict_ori_path, 'a') as predict_ori:
#                     json.dump(temp_doc, predict_ori)
#                     predict_ori.write('\n')
        
        
  # for line in open(args.model_output, 'r'):
  #   if line.strip():
  #     pred_idx = json.loads(line.strip())
  #     dia_id = pred_idx['qid'][0].split("_q#")[0]
  #     for qid, qspan, qyesno, qfollowup in zip(pred_idx['qid'], pred_idx['best_span_str'], pred_idx['yesno'], pred_idx['followup']):        
  #       preds[dia_id][qid] = qspan, qyesno, qfollowup

# output file
with open(predict_path, "w", encoding='utf8') as pred_file:
    json.dump(predict, pred_file, ensure_ascii=False)
with open(n_best_path, "w", encoding='utf8') as nb_file:
    json.dump(n_best, nb_file, ensure_ascii=False)




# {
# "best_span_str": ["In May 1983, she married Nikos Karvelas, a composer,", 
#                     "CANNOTANSWER", 
#                     "CANNOTANSWER", 
#                     "CANNOTANSWER", 
#                     "CANNOTANSWER", 
#                     "In 1986, she participated at the Cypriot National Final for Eurovision Song Contest with the song Thelo Na Gino Star (\"I Want To Be A Star\"), taking second place.", 
#                     "reached gold status selling 80.000 units.", 
#                     "In 1986 I Epomeni Kinisi (\"The Next Move\") was released. The album included the hit Pragmata (\"Things\") and went platinum, becoming the best selling record of the year."], 
# "qid": ["C_5ab583f64dbb47b995cf59328ea0af43_1_q#0", 
#         "C_5ab583f64dbb47b995cf59328ea0af43_1_q#1", 
#         "C_5ab583f64dbb47b995cf59328ea0af43_1_q#2", 
#         "C_5ab583f64dbb47b995cf59328ea0af43_1_q#3", 
#         "C_5ab583f64dbb47b995cf59328ea0af43_1_q#4", 
#         "C_5ab583f64dbb47b995cf59328ea0af43_1_q#5", 
#         "C_5ab583f64dbb47b995cf59328ea0af43_1_q#6", 
#         "C_5ab583f64dbb47b995cf59328ea0af43_1_q#7"], 
# "followup": ["y", "n", "n", "n", "n", "y", "y", "y"], 
# "yesno": ["x", "x", "x", "x", "x", "x", "x", "y"]
# }
