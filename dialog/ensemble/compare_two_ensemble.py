# -*- coding: utf8 -*-
import os, json, collections

# r1 = 'output/excord_haec10@/predictions.json'
r1 = 'output/haec10_20/predictions.json'
r2 = 'excord-ori/pretrain_model/best_model/predictions_.json'
com_path = 'compare/predictions.json'

compare = collections.OrderedDict()

with open(r1, encoding = 'utf-8') as r1_json:
    r1_file = collections.OrderedDict(json.load(r1_json))
with open(r2, encoding = 'utf-8') as r2_json:
    r2_file = collections.OrderedDict(json.load(r2_json))

r1_ave_len, r2_ave_len, r1_cna_cnt, r2_cna_cnt = 0, 0, 0, 0 
for r1_key, r1_item in r1_file.items():
    for r2_key, r2_item in r2_file.items():
        if r1_key==r2_key:
            if r1_item!=r2_item:
                r1_item = "r1_" + r1_item
                r2_item = "r2_" + r2_item
                compare[r1_key] = [r1_item, r2_item]

                if r1_item != 'CANNOTANSWER':
                    r1_ave_len = r1_ave_len + len(r1_item)
                elif r2_item != 'CANNOTANSWER':
                    r2_ave_len = r2_ave_len + len(r2_item)
                else:
                    pass

for r1_key, r1_item in r1_file.items():
    if r1_item == 'CANNOTANSWER': r1_cna_cnt = r1_cna_cnt+1
for r2_key, r2_item in r2_file.items():
    if r2_item == 'CANNOTANSWER': r2_cna_cnt = r2_cna_cnt+1

print("all q#: ", len(r1_file))     # all q#:  7354
print("com q#: ", len(compare))     # com q#:  3815

print("r1_ave_len: ", r1_ave_len/(7354-r1_cna_cnt))
print("r2_ave_len: ", r2_ave_len/(7354-r2_cna_cnt))

print(r1_cna_cnt)
print(r2_cna_cnt)

# print("r1_ave_len: ", r1_ave_len/(7354-785))
# print("r2_ave_len: ", r1_ave_len/(7354-556))

# r1_CANNOTANSWER 785
# r2_CANNOTANSWER 556
#                      51.95311310701781        17.3
# r1_ave_len(excord):  52.26259704673466        17.4
# r2_ave_len(ensemble):  50.50205942924389      16.8

with open(com_path, "w", encoding='utf8') as pred_file:
    json.dump(compare, pred_file, ensure_ascii=False)

