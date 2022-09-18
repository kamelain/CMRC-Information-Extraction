# /usr/bin/env python
# coding=utf-8
"""Evaluate the model"""
import json
import logging
import random
import argparse

from tqdm import tqdm
import os

import torch
import numpy as np
import pandas as pd

from metrics import tag_mapping_nearest, tag_mapping_corres
from utils import Label2IdxSub, Label2IdxObj
import utils
from dataloader import CustomDataLoader

# load args
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2020, help="random seed for initialization")
parser.add_argument('--ex_index', type=str, default=1)
parser.add_argument('--corpus_type', type=str, default="QuAC_test", help="NYT, WebNLG, NYT*, WebNLG*")
parser.add_argument('--device_id', type=int, default=0, help="GPU index")
parser.add_argument('--restore_file', default='last', help="name of the file containing weights to reload")

parser.add_argument('--corres_threshold', type=float, default=0.5, help="threshold of global correspondence")
parser.add_argument('--rel_threshold', type=float, default=0.5, help="threshold of relation judgement")
parser.add_argument('--ensure_corres', action='store_true', help="correspondence ablation")
parser.add_argument('--ensure_rel', action='store_true', help="relation judgement ablation")
parser.add_argument('--emb_fusion', type=str, default="concat", help="way to embedding")
parser.add_argument('--mode', type=str, default="app", help="data_sign")

def get_metrics(correct_num, predict_num, gold_num):
    p = correct_num / predict_num if predict_num > 0 else 0
    r = correct_num / gold_num if gold_num > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {
        'correct_num': correct_num,
        'predict_num': predict_num,
        'gold_num': gold_num,
        'precision': p,
        'recall': r,
        'f1': f1
    }


def span2str(triples, tokens):
    def _concat(token_list):
        result = ''
        for idx, t in enumerate(token_list):
            if idx == 0:
                result = t
            elif t.startswith('##'):
                result += t.lstrip('##')
            else:
                result += ' ' + t
        return result

    output = []
    for triple in triples:
        rel = triple[-1]
        sub_tokens = tokens[triple[0][1]:triple[0][-1]]
        obj_tokens = tokens[triple[1][1]:triple[1][-1]]
        sub = _concat(sub_tokens)
        obj = _concat(obj_tokens)
        output.append((sub, obj, rel))
    return output


def evaluate(model, data_iterator, params, ex_params, mark='App'):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()
    # rel_num = params.rel_num

    predictions = []
    # ground_truths = []
    # correct_num, predict_num, gold_num = 0, 0, 0

    for batch in tqdm(data_iterator, unit='Batch', ascii=True):
        # to device
        batch = tuple(t.to(params.device) if isinstance(t, torch.Tensor) else t for t in batch)
        # input_ids, attention_mask, triples, input_tokens = batch
        input_ids, attention_mask, input_tokens = batch
        bs, seq_len = input_ids.size()

        # inference
        with torch.no_grad():
            ###
            pred_seqs, pre_corres, xi, pred_rels = model(input_ids, attention_mask=attention_mask,
                                                         ex_params=ex_params)
            
            # print("pred_seqs: ", pred_seqs)
            # print("pre_corres: ", pre_corres)
            # print("xi: ", xi)
            # print("pred_rels: ", pred_rels)

            # (sum(x_i), seq_len)
            pred_seqs = pred_seqs.detach().cpu().numpy()
            # (bs, seq_len, seq_len)
            pre_corres = pre_corres.detach().cpu().numpy()
        if ex_params['ensure_rel']:
            # (bs,)
            xi = np.array(xi)
            # (sum(s_i),)
            pred_rels = pred_rels.detach().cpu().numpy()
            # decode by per batch
            xi_index = np.cumsum(xi).tolist()
            # (bs+1,)
            xi_index.insert(0, 0)

        for idx in range(bs):
            # if ex_params['ensure_rel']:
            pre_triples = tag_mapping_corres(predict_tags=pred_seqs[xi_index[idx]:xi_index[idx + 1]],
                                                pre_corres=pre_corres[idx],
                                                pre_rels=pred_rels[xi_index[idx]:xi_index[idx + 1]],
                                                label2idx_sub=Label2IdxSub,
                                                label2idx_obj=Label2IdxObj)
            predictions.append(list(set(pre_triples)))
    return predictions, xi

def output(predictions, xi):
    for idx, item in enumerate(predictions):
        item.append(xi[idx])
    return predictions


if __name__ == '__main__':
    args = parser.parse_args()
    params = utils.Params(ex_index=args.ex_index, corpus_type=args.corpus_type)
    ex_params = {
        # 'corres_threshold': args.mat_threshold,
        # 'rel_threshold': args.rel_pre_threshold,

        # 'ensure_corres': args.ensure_match,
        # 'ensure_rel': args.ensure_relpre,
        # 'emb_fusion': args.emb_fusion
        'ensure_corres': args.ensure_corres,
        'ensure_rel': args.ensure_rel,
        'emb_fusion': args.emb_fusion
    }

    torch.cuda.set_device(args.device_id)
    print('current device:', torch.cuda.current_device())
    mode = args.mode
    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed

    # Set the logger
    utils.set_logger()

    # get dataloader
    dataloader = CustomDataLoader(params)

    # Define the model
    logging.info('Loading the model...')
    logging.info(f'Path: {os.path.join(params.model_dir, args.restore_file)}.pth.tar')
    # Reload weights from the saved file
    model, optimizer = utils.load_checkpoint(os.path.join(params.model_dir, args.restore_file + '.pth.tar'))
    model.to(params.device)
    # logging.info('- done.')

    logging.info("Loading the dataset...")
    loader= dataloader.get_dataloader(data_sign=mode, ex_params=ex_params)
    # logging.info('-done')

    logging.info("Starting prediction...")
    predictions, xi = evaluate(model, loader, params, ex_params, mark=mode)
    # print("predictions: ", predictions)
    # print("xi: ", xi)
    # result = output(predictions, xi)
    # print(result)

    with open("data/QuAC_test/app_triples.json", encoding = 'utf-8') as f0:
        dataset = list(json.load(f0))
    
    # for idx, id in enumerate(ids):
    #     for item in dataset:
    #         if id==item['id']:
    #             item['triples']=predictions[idx]
    for idx,item in enumerate(dataset):
        # triples = list(predictions[idx])
        triples = predictions[idx]
        a = []
        trans_triples = []
        # print("triples type: ", type(triples))
        if triples:     # if triples not empty
            for a in triples:
                a = [list(a[0]), list(a[1]), a[2].item()]
                trans_triples.append(a)
                # a:  (('H', 16, 17), ('T', 64, 68), 7)  ,type(a): <class 'tuple'>
                # a[0]:  ('H', 16, 17)  ,type(a[0]): <class 'tuple'>
                # a[1]:  ('T', 64, 68)  ,type(a[1]): <class 'tuple'>
                # a[2]:  7  ,type(a[2]): <class 'numpy.int64'>
        else: trans_triples.append(triples)

        item['triples'] = trans_triples
                    
    with open("data/QuAC_test/app_triples_result.json", 'w') as f:
        json.dump(dataset, f)

    logging.info('-done')
