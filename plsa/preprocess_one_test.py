q_new, q_list, q_done_list = '', [], []
q = 'what is clarinet count?' 
plsa_lists = [['basie', 'clarinet'], ['count', 'prominence']]

print("q: ", q)

for wd_q in q.split():
    print("# wd_q", wd_q) 
    q_new = q_new + " " + wd_q
    for plsa_list in plsa_lists:
        print("  plsa_list: ", plsa_list)
        for idx, wd_pl in enumerate(plsa_list):
            print("  -> wd_pl", wd_pl)
            if wd_q.casefold()==wd_pl.casefold():
                print(" ### same, wd_q/wd_pl: ", wd_pl)
                # print("    wd_q not in q_done_list? ", wd_q not in q_done_list)
            # if wd_q not in q_done_list and wd_q.casefold()==wd_pl.casefold() and wd_pl not in q.split(): 
            if wd_q not in q_done_list and wd_q.casefold()==wd_pl.casefold():
                if idx==0: wd_kg = plsa_list[1]
                elif idx==1: wd_kg = plsa_list[0]
                print("kg[0]: ", plsa_list[0], "; kg[1]: ", plsa_list[1])
                print("wd_kg: ", wd_kg)
                q_new = q_new + " " + wd_kg
                q_list.append(wd_kg)                # [basie, prominence]       -> extend kg
                q_done_list.append(wd_q)            # ['clarinet', 'count']     -> entity

print('qas["question_plsa"]: ', q_new)
print('qas["question_plsa_list"]: ', q_list)
