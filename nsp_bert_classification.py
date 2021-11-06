#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge_sy"
# Date: 2021/6/30

import numpy
from tqdm import tqdm
from sklearn import metrics
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding, DataGenerator
from utils import *
from hyper_parameters import *

class data_generator(DataGenerator):
    """Data Generator"""
    def __init__(self, pattern="", is_pre=True, *args, **kwargs):
        super(data_generator, self).__init__(*args, **kwargs)
        self.pattern = pattern
        self.is_pre = is_pre
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text, label) in self.sample(random):
            if (self.is_pre):
                token_ids, segment_ids = tokenizer.encode(first_text=self.pattern, second_text=text, maxlen=maxlen)
            else:
                token_ids, segment_ids = tokenizer.encode(first_text=text, second_text=self.pattern, maxlen=maxlen)
            source_ids, target_ids = token_ids[:], token_ids[:]
            batch_token_ids.append(source_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(target_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_output_ids = sequence_padding(batch_output_ids)
                yield [batch_token_ids, batch_segment_ids, batch_output_ids], None
                batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []

def evaluate(data_generator_list, data, note=""):
    print("\n*******************Start to Zero-Shot predict on 【{}】*******************".format(note), flush=True)
    patterns_logits = [[] for _ in patterns]
    for i in range(len(data_generator_list)):
        print("\nPattern{}".format(i), flush=True)
        data_generator = data_generator_list[i]
        counter = 0
        for (x, _) in tqdm(data_generator):
            outputs = model.predict(x[:2])
            for out in outputs:
                logit_pos = out[0].T
                patterns_logits[i].append(logit_pos)
                counter += 1

    # Evaluate the results
    trues = [d[1] for d in data]
    preds = []
    for i in range(len(patterns_logits[0])):
        pred = numpy.argmax([logits[i] for logits in patterns_logits])
        preds.append(int(pred))

    confusion_matrix = metrics.confusion_matrix(trues, preds, labels=None, sample_weight=None)
    print("Confusion Matrix:\n{}".format(confusion_matrix), flush=True)
    if (dataset.metric == 'Matthews'):
        matthews_corrcoef = metrics.matthews_corrcoef(trues, preds)
        print("Matthews Corrcoef:\n{}".format(matthews_corrcoef), flush=True)
    else:
        acc = metrics.accuracy_score(trues, preds, normalize=True, sample_weight=None)
        print("Acc.:\t{:.4f}".format(acc), flush=True)
        return acc

if __name__ == "__main__":

    # Load the hyper-parameters-----------------------------------------------------------
    maxlen = 256  # The max length 128 is used in our paper
    batch_size = 40  # Will not influence the results

    # Choose a dataset----------------------------------------------------------------------
    # For Chinese datasets
    # dataset_names = ['eprstmt', 'tnews', 'csldcp', 'iflytek']
    # For English datasets in KPT
    # dataset_names = ['AGNews', 'DBPedia', 'IMDB', 'Amazon']
    # For GLUE benchmark
    # dataset_names = ['CoLA', 'SST-2']
    # Others in LM-BFF
    # dataset_names = ['SST-5', 'MR', 'CR', 'MPQA', 'Subj', 'TREC']
    dataset_name = 'SST-2'

    # Choose a model----------------------------------------------------------------------
    # Recommend to use 'uer-mixed-bert-base' and 'google-bert-cased'
    # model_names = ['google-bert', 'google-bert-small', 'google-bert-cased',
    #                'google-bert-wwm-large', 'google-bert-cased-wwm-large',
    #                'google-bert-zh', 'hfl-bert-wwm', 'hfl-bert-wwm-ext',
    #                'uer-mixed-bert-tiny', 'uer-mixed-bert-small',
    #                'uer-mixed-bert-base', 'uer-mixed-bert-large']
    model_name = MODEL_NAME[dataset_name]

    # Load model and dataset class
    bert_model = Model(model_name=model_name)
    dataset = Datasets(dataset_name=dataset_name)

    # Choose a template [0, 1, 2]--------------------------------------------------------
    patterns = dataset.patterns[PATTERN_INDEX[dataset_name]]

    # Prefix or Suffix-------------------------------------------------------------------
    is_pre = IS_PRE[dataset_name]

    # Load the dev set--------------------------------------------------------------------
    # -1 for all the samples
    dev_data = dataset.load_data(dataset.dev_path, sample_num=-1, is_shuffle=True)
    dev_generator_list = []
    for p in patterns:
        dev_generator_list.append(data_generator(pattern=p, is_pre=is_pre, data=dev_data, batch_size=batch_size))

    # Load the test set--------------------------------------------------------------------
    # -1 for all the samples
    test_data = dataset.load_data(dataset.test_path, sample_num=-1, is_shuffle=True)
    test_generator_list = []
    for p in patterns:
        test_generator_list.append(data_generator(pattern=p, is_pre=is_pre, data=test_data, batch_size=batch_size))

    # Build BERT model---------------------------------------------------------------------
    tokenizer = Tokenizer(bert_model.dict_path, do_lower_case=True)
    # Load BERT model with NSP head
    model = build_transformer_model(
        config_path=bert_model.config_path,
        checkpoint_path=bert_model.checkpoint_path,
        with_nsp=True,
    )

    # Zero-Shot predict and evaluate-------------------------------------------------------
    evaluate(dev_generator_list, dev_data, note="Dev Set")
    evaluate(test_generator_list, test_data, note="Test Set")
