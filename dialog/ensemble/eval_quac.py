import json, string, re, os
from collections import Counter, defaultdict
from argparse import ArgumentParser
from scorer import external_call 

# predict target
output_file = "haec/bert_out/bg_10/predictions_24000.json"
# output_file = "ex2hae/best.json"

val_file = "cls/quac_bg/QuAC10/val_v0.2.json"
# output_file = "output/excord_haec10/predictions_ori.json"
# output_file = "excord/output/result/predictions_.json"
# result_path = "output/excord_haec10/result.json"

val_file_json = json.load(open(val_file, 'r'))['data']
val_eval_res = external_call(val_file_json, output_file)

val_f1 = val_eval_res['f1']
val_followup = val_eval_res['followup']
val_yesno = val_eval_res['yes/no']
val_heq = val_eval_res['HEQ']
val_dheq = val_eval_res['DHEQ']

print("val_eval_res: ", val_eval_res)
print("val_f1", val_f1) 
# with open(result_path, 'w') as result:
#     json.dump(val_eval_res, result)
