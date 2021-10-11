**[English](https://github.com/sunyilgdx/NSP-BERT)** | **[中文](https://github.com/sunyilgdx/NSP-BERT/blob/main/README_zh.md)**

## 概要

这是我们论文 **[NSP-BERT: A Prompt-based Zero-Shot Learner Through an Original Pre-training Task —— Next Sentence Prediction](https://arxiv.org/abs/2109.03564)** 的源码. 我们利用了一个 **句子级别(sentence-level)** 的预训练任务 **NSP (下一句预测，Next Sentence Prediction)** 来实现不同的NLP下游任务, 例如 *单句分类(single sentence classification)*, *双句分类(sentence pair classification)*, *指代消解(coreference resolution)*, *完形填空(cloze-style task)*, *实体链接(entity linking)*, *实体类型识别(entity typing)*.

在[FewCLUE benchmark](https://github.com/CLUEbenchmark/FewCLUE)评测集上的部分任务中, 我们的 NSP-BERT性能远超其他 zero-shot 方法 (GPT-1-zero and PET-zero). 我们希望 NSP-BERT 能够成为一个可以辅助其他语言模型的一个无监督工具.

## 新闻

2021/10/11 我们上传了几个英文分类任务的代码，**AG’s News, DBPedia, Amazon and IMDB**，十分感谢[Shengding Hu](https://github.com/ShengdingHu)和他的[KnowledgeablePromptTuning](https://github.com/ShengdingHu/KnowledgeablePromptTuning).


## 目录
| 章节 | 描述 |
|-|-|
| [开发环境](#开发环境)   | 开发环境 |
| [下载](#下载)           | NSP-BERT使用模型的下载方式 |
| [演示样例](#演示样例)   | 中文和英文Demo |
| [评测方法](#评测方法)   | 对NSP-BERT进行评测的代码 |
| [基线模型](#基线模型)   | 基线模型介绍 |
| [模型比较](#模型比较)   | 模型评测结果比较 |
| [策略细节](#策略细节)   | 不同任务的策略 |
| [探讨展望](#探讨展望)   | 对论文的探讨和展望 |
| [鸣谢](#鸣谢)           | 鸣谢苏神|
 
## 开发环境
开发环境如下所示:
```
Python 3.6
bert4keras 0.10.6
tensorflow-gpu 1.15.0
```

## 下载
### 模型下载
需要下载不同预训练模型的checkpoints.  *vocab.txt* 和 *config.json* 已经在我们的仓库里了 [repository](https://github.com/sunyilgdx/NSP-BERT/tree/main/models).

| 发布组织                                          | 模型名称         | 模型参数              | 下载链接                                                                               | Tips |
|---------------------------------------------------|------------------|-----------------------|-------------------------------------------------------------------------------------------------|------|
| [Google](https://github.com/google-research/bert) | BERT-uncased     | L=12 H=769 A=12 102M  | [Tensorflow](https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip) |      |
|                                                   | BERT-Chinese     | L=12 H=769 A=12 102M  | [Tensorflow](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) |      |
| [HFL](https://github.com/ymcui/Chinese-BERT-wwm)  | BERT-wwm         | L=12 H=769 A=12 102M  | [Tensorflow](https://pan.iflytek.com/link/A2483AD206EF85FD91569B498A3C3879)|  |
|                                                   | BERT-wwm-ext     | L=12 H=769 A=12 102M  | [Tensorflow](https://pan.iflytek.com/link/A2483AD206EF85FD91569B498A3C3879)|  |
| [UER](https://github.com/dbiir/UER-py)            | BERT-mixed-tiny  | L=3  H=384 A=6  14M   | [Pytorch](https://share.weiyun.com/yXx0lfUg)  | \* |
|                                                   | BERT-mixed-Small | L=6  H=512 A=8  31M   | [Pytorch](https://share.weiyun.com/fhcUanfy)  | \* |
|                                                   | BERT-mixed-Base  | L=12 H=769 A=12 102M  | [Pytorch](https://share.weiyun.com/5QOzPqq)   | \* |
|                                                   | BERT-mixed-Large | L=24 H=1024 A=16 327M | [Pytorch](https://share.weiyun.com/5G90sMJ)   | \* |

\* 我们需要使用 [UER的转换工具](https://github.com/dbiir/UER-py/blob/master/scripts/convert_bert_from_uer_to_original_tf.py) 将 UER pytorch 模型转换成 Original Tensorflow.

### 数据集下载
在论文的实验部分，我们使用 FewCLUE 评测集和 DuEL2.0 (CCKS2020).
| 数据集             | 下载链接                                                    |
|--------------------|-------------------------------------------------------------|
| FewCLUE            | https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets |
| DuEL2.0 (CCKS2020) | https://aistudio.baidu.com/aistudio/competition/detail/83   |
| enEval             | https://github.com/ShengdingHu/KnowledgeablePromptTuning    |

将数据集放在 [NSP-BERT/datasets/]()下.

## 演示样例
尝试使用 *./demos/nsp_bert_classification_demo.py* and *./demos/nsp_bert_classification_demo_en.py* 来完成分类任务.
编辑自己的 **Labels** 和 **Samples**, 创造 **Prompt Templates**, 就可以进行文本你分类了.

```
...
label_names = ['娱乐', '体育', '音乐', '电竞', '经济', '教育']
patterns = ["这是一篇{}新闻".format(label) for label in label_names]
demo_data_zh = ['梅西超越贝利成为南美射手王',
                 '贾斯汀比伯发布新单曲',
                 '比心APP被下架并永久关闭陪玩功能',
                 '徐莉佳的伦敦奥运金牌氧化了',
                 '10元芯片卖400元!芯片经销商被罚',
                 '北京首批校外培训机构白名单公布']
...
```
**输出**
```
Sample 0:
Original Text: 梅西超越贝利成为南美射手王
Predict label: 体育
Logits: [0.67886037, 0.98553574, 0.16819017, 0.6733272, 0.29652277, 0.07275329]

Sample 1:
Original Text: 贾斯汀比伯发布新单曲
Predict label: 音乐
Logits: [0.95801944, 0.4572674, 0.9918983, 0.35939765, 0.3782271, 0.12813713]

Sample 2:
Original Text: 比心APP被下架并永久关闭陪玩功能
Predict label: 娱乐
Logits: [0.40367377, 0.23919956, 0.1673808, 0.20248286, 0.29829133, 0.122355804]
...
```



## 评测方法
通过运行不同的 python 文件对 NSP-BERT 进行评测.

```
NSP-BERT
    |- datasets
        |- clue_datasets
           |- ...
        |- DuEL 2.0
           |- dev.json
           |- kb.json
        |- enEval
           |- agnews
           |- amazon
           |- dbpedia
           |- imdb
    |- demos
        |- nsp_bert_classification_demo.py
        |- nsp_bert_classification_demo_en.py
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

| Python 文件                            | 任务                                 | 数据集                             |
|----------------------------------------|--------------------------------------|------------------------------------|
| [nsp_bert_classification.py]()         | **Single Sentence Classification**   | *EPRSTMT, TNEWS, CSLDCP, IFLYTEK*  |
|                                        |                                      | *AG’s News, DBPedia, Amazon, IMDB* |
| [nsp_bert_sentence_pair.py]()          | **Sentence Pair Classification**     | *OCNLI, BUSTM, CSL*                |
| [nsp_bert_cloze_style.py]()            | **Cloze-style Task**                 | *ChID*                             |
| [nsp_bert_coreference_resolution.py]() | **Coreference Resolution**           | *CLUEWSC*                          |
| [nsp_bert_entity_linking.py]()         | **Entity Linking and Entity Typing** | *DuEL2.0*                          |

## 基线模型

参考 FewCLUE, 我们选择了3个场景, fine-tuning, few-shot and zero-shot.
对于机型模型我们采用 Chineses-RoBERTa-Base 和 Chinses-GPT-1 作为骨干预训练模型.

### 算法
| 场景            | 算法                                  |
|-----------------|---------------------------------------|
| **Fine-tuning** | *BERT, RoBERTa*                       |
| **Few-Shot**    | *PET, ADAPET,  P-tuning, LM-BFF, EFL* |
| **Zero-Shot**   | *GPT-zero, PET-zero*                  |

### 下载

| 发布组织                                                                                                 | 模型名称                | 模型参数             | 下载链接                                                           |
|----------------------------------------------------------------------------------------------------------|-------------------------|----------------------|-----------------------------------------------------------------------------|
| [huawei-noah](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-Gen-TensorFlow) | Chinese GPT             | L=12 H=769 A=12 102M | [Tensorflow](https://pan.baidu.com/s/1Bgle8TpcxHyuUz_jAXOBWw)               |
| [HFL](https://github.com/ymcui/Chinese-BERT-wwm)                                                         | RoBERTa-wwm-ext | L=12 H=769 A=12 102M | [Tensorflow](https://pan.iflytek.com/link/98D11FAAF0F0DBCB094EE19CCDBC98BF) |



## 模型比较

<br/><img src="./images/main_results.png" width="800"  alt="Main Results"/><br/>

## 策略细节

<br/><img src="./images/strategies.png" width="600"  alt="Strategies"/><br/>


## 探讨展望

* 由于 NSP-BERT 是一个句子级的 prompt-learning 模型, 相比于 GPT-zero and PET-zero, 其在 **Single Sentence Classification** 等任务上 (*TNEWS, CSLDCP and IFLYTEK*)有着显著的提高. 同时, 可以很好地完成 **实体链接** 任务 (*DuEL2.0*), 且不受限于不同长度地实体描述， 这是 GPT-zero 和 PET-zero 所不能做到的.
* 但是, 其在 **词级别** 的任务上, 例如 **完形填空** 和 **实体类别识别** 上效果一般.
* 在将来的工作中, 可以继续将其应用在Few-Shot场景中.

## 鸣谢

* 我们的代码基于[苏剑林](https://github.com/bojone)老师的[bert4keras](https://github.com/bojone/bert4keras)开源项目.
* 感谢苏剑林老师, 他的系列博客[科学空间](https://kexue.fm/), 以及他的开源精神, 启发和激励了我的论文写作过程.

## 引用
```
@misc{sun2021nspbert,
    title={NSP-BERT: A Prompt-based Zero-Shot Learner Through an Original Pre-training Task--Next Sentence Prediction},
    author={Yi Sun and Yu Zheng and Chao Hao and Hangping Qiu},
    year={2021},
    eprint={2109.03564},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
