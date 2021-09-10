## Overview

This is the code of our paper **[NSP-BERT: A Prompt-based Zero-Shot Learner Through an Original Pre-training Task —— Next Sentence Prediction](https://arxiv.org/abs/2109.03564)**. We use a sentence-level pre-training task NSP (Next Sentence Prediction) to realize prompt-learning and perform various downstream tasks, such as *single sentence classification*, *sentence pair classification*, *coreference resolution*, *cloze-style task*, *entity linking*, *entity typing*.

On the [FewCLUE benchmark](https://github.com/CLUEbenchmark/FewCLUE), our NSP-BERT outperforms other zero-shot methods (GPT-1-zero and PET-zero) on most of these tasks and comes close to the few-shot methods. We hope NSP-BERT can be an unsupervised tool that can assist other language tasks or models.


## Guide
| Section | Description |
|-|-|
| [Environment](#Environment) | The required deployment environment |
| [Downloads](#Downloads) | Download links for the models' checkpoints used by NSP-BERT |
| [Use examples](#Use-examples) | Learn to use NSP-BERT for different downstream tasks |
| [Baselines](#Baselines) | Baseline results for several Chinese NLP datasets (partial) |
| [Model Comparison](#Model-Comparison) | Compare the models published in this repository |
| [Strategy Details](#Strategy-Details) | Some of the strategies used in the paper |
| [Discussion](#Discussion) | Discussion and Discrimination for future work |
 
## Environment
The environments are as follows:
```
Python 3.6
bert4keras 0.10.6
tensorflow-gpu 1.15.0
```

## Downloads
### Models
We should dowmload the checkpoints of different models. The *vocab.txt* and the *config.json* are already in our [repository](https://github.com/sunyilgdx/NSP-BERT/tree/main/models).

| Organization                                      | Model Name       | Model Parameters      | Download Linking                                                                                | Tips |
|---------------------------------------------------|------------------|-----------------------|-------------------------------------------------------------------------------------------------|------|
| [Google](https://github.com/google-research/bert) | BERT-Chinese     | L=12 H=769 A=12 102M  | [Tensorflow](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) |      |
| [HFL](https://github.com/ymcui/Chinese-BERT-wwm)  | BERT-wwm         | L=12 H=769 A=12 102M  | [Tensorflow](https://pan.iflytek.com/link/A2483AD206EF85FD91569B498A3C3879)|  |
|                                                   | BERT-wwm-ext     | L=12 H=769 A=12 102M  | [Tensorflow](https://pan.iflytek.com/link/A2483AD206EF85FD91569B498A3C3879)|  |
| [UER](https://github.com/dbiir/UER-py)            | BERT-mixed-tiny  | L=3  H=384 A=6  14M   | [Pytorch](https://share.weiyun.com/yXx0lfUg)  | \* |
|                                                   | BERT-mixed-Small | L=6  H=512 A=8  31M   | [Pytorch](https://share.weiyun.com/fhcUanfy)  | \* |
|                                                   | BERT-mixed-Base  | L=12 H=769 A=12 102M  | [Pytorch](https://share.weiyun.com/5QOzPqq)   | \* |
|                                                   | BERT-mixed-Large | L=24 H=1024 A=16 327M | [Pytorch](https://share.weiyun.com/5G90sMJ)   | \* |

\* We need to use [UER's convert tool](https://github.com/dbiir/UER-py/blob/master/scripts/convert_bert_from_uer_to_original_tf.py) to convert UER pytorch to Original Tensorflow.

### Datasets
We use FewCLUE datasets and DuEL2.0 (CCKS2020) in our experiments.
| Datasets           | Download Links                                              |
|--------------------|-------------------------------------------------------------|
| FewCLUE            | https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets |
| DuEL2.0 (CCKS2020) | https://aistudio.baidu.com/aistudio/competition/detail/83   |

Put the datasets into the [NSP-BERT/datasets/](https://github.com/sunyilgdx/NSP-BERT/tree/main/datasets/).

## Use examples
We can run individual python files in the project directly to evaluate our NSP-BERT.

```
NSP-BERT
    |- datasets
        |- clue_datasets
           |- ...
        |- DuEL 2.0
           |- dev.json
           |- kb.json
    |- models
        |- uer_mixed_corpus_bert_base
           |- bert_config.json
           |- vocab.txt
           |- bert_model.ckpt...
           |- ...
    |- nsp_bert_classification.py             # Single Sentence Classification
    |- nsp_bert_sentence_pair.py              # Sentence Pair Classification
    |- nsp_bert_cloze_style.py                # Cloze-style Task
    |- nsp_bert_coreference_resolution.py     # Coreference Resolution
    |- nsp_bert_entity_linking.py             # Entity Linking and Entity Typing
    |- utils.py
```

| Python File                            | Task                                 | Datasets                          |
|----------------------------------------|--------------------------------------|-----------------------------------|
| [nsp_bert_classification.py]()         | **Single Sentence Classification**   | *EPRSTMT, TNEWS, CSLDCP, IFLYTEK* |
| [nsp_bert_sentence_pair.py]()          | **Sentence Pair Classification**     | *OCNLI, BUSTM, CSL*               |
| [nsp_bert_cloze_style.py]()            | **Cloze-style Task**                 | *ChID*                            |
| [nsp_bert_coreference_resolution.py]() | **Coreference Resolution**           | *CLUEWSC*                         |
| [nsp_bert_entity_linking.py]()         | **Entity Linking and Entity Typing** | *DuEL2.0*                         |

## Baselines

Reference FewCLUE, we choos 3 training scenarios, fine-tuning, few-shot and zero-shot.
The baselines use Chineses-RoBERTa-Base and Chinses-GPT-1 as the backbone model.

### Methods
| Scenarios       | Methods                               |
|-----------------|---------------------------------------|
| **Fine-tuning** | *BERT, RoBERTa*                       |
| **Few-Shot**    | *PET, ADAPET,  P-tuning, LM-BFF, EFL* |
| **Zero-Shot**   | *GPT-zero, PET-zero*                  |

### Downloads

| Organization                                                                                             | Model Name              | Model Parameters     | Download Linking                                                            |
|----------------------------------------------------------------------------------------------------------|-------------------------|----------------------|-----------------------------------------------------------------------------|
| [huawei-noah](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-Gen-TensorFlow) | Chinese GPT             | L=12 H=769 A=12 102M | [Tensorflow](https://pan.baidu.com/s/1Bgle8TpcxHyuUz_jAXOBWw)               |
| [HFL](https://github.com/ymcui/Chinese-BERT-wwm)                                                         | RoBERTa-wwm-ext | L=12 H=769 A=12 102M | [Tensorflow](https://pan.iflytek.com/link/98D11FAAF0F0DBCB094EE19CCDBC98BF) |



## Model Comparison

<br/><img src="./images/main_results.png" width="800"  alt="Main Results"/><br/>

## Strategy Details

<br/><img src="./images/strategies.png" width="800"  alt="Strategies"/><br/>


## Discussion

* Sincce NSP-BERT is a sentence-level prompt-learning model, it is significantly superior to GPT-zero and PET-zero in terms of **Single Sentence Classification** tasks (*TNEWS, CSLDCP and IFLYTEK*). At the same time, it can solve the **Entity Linking** task (*DuEL2.0*), and the model is not limited by the non-fixed-length entity description, which GPT-zero and PET-zero cannot do this.
* However, it doesn't work as well on **Token-Level** tasks, such as **Cloze-style** task and **Entity Typing**.
* In future work, it is essential to extend NSP-BERT to the few-shot scenario.
