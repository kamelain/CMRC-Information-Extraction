# CMRC-Information-Extraction-and-Knowledge-Injection
Multi-turn MRC Implementation for "A Study on the Information Extraction and Knowledge Injection for Machine Reading Comprehension"


基於 [BERT](https://github.com/google-research/bert)、[BERT-HAE](https://github.com/prdwb/bert_hae)、[BERT-HAM](https://github.com/prdwb/attentive_history_selection) 以及 [ExCorD](https://github.com/dmis-lab/excord) 的資源修改。
適用於英文多輪機器閱讀理解 [QuAC](https://quac.ai)。

本程式碼為論文 [A Study on the Information Extraction and Knowledge Injection for Machine Reading Comprehension](https://etds.lib.ntnu.edu.tw/thesis/detail/c7f11bb51318d02b9874ae5429b6eb82/?seq=1) 於多輪機器閱讀理解的實作部分。實作分為四個部分。

>(1) Baseline: BERT, BERT-HAE, ExCorD

>(2) Information Extraction（資訊擷取）

>(3) Knowledge Graph（知識注入）

>(4) Ensemble（N-best 答案進行 Reranking）



>Clutering Strategies 
![Clustering Strategies](https://github.com/kamelain/CMRC-Information-Extraction/blob/main/Screen%20Shot%202022-09-19%20at%2012.49.15%20AM.png)


>Information Extraction 架構
![IE](https://github.com/kamelain/CMRC-Information-Extraction/blob/main/Screen%20Shot%202022-09-19%20at%2012.49.29%20AM.png)


>Knowledge Injection - WortNet 架構
![KI1](https://github.com/kamelain/CMRC-Information-Extraction/blob/main/Screen%20Shot%202022-09-19%20at%2012.49.48%20AM.png)


>Knowledge Injection - PLSA/PRGC 架構
![KI2](https://github.com/kamelain/CMRC-Information-Extraction/blob/main/Screen%20Shot%202022-09-19%20at%2012.49.40%20AM.png)


>Answer Re-ranking 示意圖
![reranking](https://github.com/kamelain/CMRC-Information-Extraction/blob/main/Screen%20Shot%202022-09-19%20at%2012.50.26%20AM.png)



### Conda 環境設定檔

>* tf.yml 模型訓練環境 (BERT-HAE, BERT-HAM)
>* ex.yml 模型訓練環境 (ExCorD-BERT, ExCorD-RoBERTa, PRGC, K-BERT)
>* ex2.yml 模型訓練環境 (ExCorD-DeBERTa)
>* prep.yml 資料預處理環境 (Clustering, Ensemble, WordNet, PLSA)


### 主程式位置

>* BERT-HAE: dialog/hea_clustering/bert_hae-master
>* BERT-HAE + IE: dialog/hea_clustering/hae_kg
>* BERT-HAE + KI: dialog/hea_clustering/hae_kg
>* ExCorD: dialog/excord/excord-main
>* ExCorD + IE: dialog/excord/excord-clus
>* ExCorD + KI: dialog/excord/kbert-ex
>* Ensemble: dialog/ensemble

>* BERT-HAM + IE: dialog/attentive_cls
>* PRGC-based KI: dialog/PRGC-main

>* WordNet based KG: dialog/PRGC-main
>* PLSA-based KG: plsa
>* PRGC based KG: wordnet


### 使用

* Information Extraction

```
bash cls/run_cls.sh
```

* Train & Prediction

```
bash run.sh
```

* Evaluate

```
bash eval.sh
```


>Information Extraction 結果
![Result-IE](https://github.com/kamelain/CMRC-Information-Extraction/blob/main/Screen%20Shot%202022-09-19%20at%2012.51.09%20AM.png)

>Knowledge Injection 結果
![Result-KI](https://github.com/kamelain/CMRC-Information-Extraction/blob/main/Screen%20Shot%202022-09-19%20at%2012.51.13%20AM.png)
