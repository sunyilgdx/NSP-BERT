#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge_sy"
# Date: 2021/6/30


from tqdm import tqdm
from sklearn import metrics
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding, DataGenerator
from utils import *
import time

class data_generator(DataGenerator):
    """数据生成器"""

    def __init__(self, is_pre=True, is_rel_pos=False, pattern='', *args, **kwargs):
        super(data_generator, self).__init__(*args, **kwargs)
        self.is_pre = is_pre
        # self.is_rel_pos = is_rel_pos
        self.pattern = pattern

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text, label, span1_text, span2_text, span1_index, span2_index) in self.sample(random):
            text = "{}{}{}是指".format(text, self.pattern, span2_text)
            if (self.is_pre):
                token_ids, segment_ids = tokenizer.encode(first_text=span1_text, second_text=text, maxlen=maxlen)
            else:
                token_ids, segment_ids = tokenizer.encode(first_text=text, second_text=span1_text, maxlen=maxlen)

            source_ids, target_ids = token_ids[:], token_ids[:]
            batch_token_ids.append(source_ids)
            batch_segment_ids.append(segment_ids)

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids, = [], []

def evaluate_dev(data_generator, data, label_num, note=""):

    print("\n*******************Start to Zero-Shot predict on 【{}】*******************".format(note), flush=True)
    logits = []
    id2logit = {}

    id = 0
    for (x, _) in tqdm(data_generator):
        outputs = model.predict(x)
        for out in outputs:
            logit_pos = out[0].T
            logits.append(logit_pos)
            id2logit[id] = logit_pos
            id += 1

    # Evaluate the results
    thresholds = [0] * (label_num - 1)
    trues = [d[1] for d in data]
    preds = [0] * len(data)
    sorted_id_logit = sorted(id2logit.items(), key=lambda item: item[1])
    split_num = len(data) // label_num
    start = 0

    for i in range(label_num):
        end = start + split_num if i != label_num - 1 else len(data)
        logit = - 1.
        for (id, logit) in sorted_id_logit[start: end]:
            preds[id] = i
        if (i < label_num - 1):
            thresholds[i] = logit
        start = end

    confusion_matrix = metrics.confusion_matrix(trues, preds, labels=None, sample_weight=None)
    acc = metrics.accuracy_score(trues, preds, normalize=True, sample_weight=None)
    print("Confusion Matrix:\n{}".format(confusion_matrix), flush=True)
    print("Acc.:\t{:.4f}".format(acc), flush=True)
    return acc, thresholds

def evaluate_test_batch(data_generator, data, label_num, note):
    print("\n*******************Start to Zero-Shot predict on 【{}】 with different batch sizes*******************".format(note), flush=True)
    logits = []
    id2logit = {}
    id = 0
    for (x, _) in tqdm(data_generator):
        outputs = model.predict(x)
        for out in outputs:
            logit_pos = out[0].T
            logits.append(logit_pos)
            id2logit[id] = logit_pos
            id += 1

    # Evaluate the results
    trues = [d[1] for d in data]
    preds = [0] * len(data)
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, len(data)]
    batch_acc_list = []
    for batch_size in batch_sizes:
        sub_id2logit_list = divide_dict_batch(id2logit, batch_size)
        for sub_id2logit in sub_id2logit_list:
            sorted_id_logit = sorted(sub_id2logit.items(), key=lambda item: item[1])
            split_num = len(sub_id2logit) // label_num
            start = 0
            for i in range(label_num):
                end = start + split_num if i != label_num - 1 else len(sub_id2logit)
                for (id, logit) in sorted_id_logit[start: end]:
                    preds[id] = i
                start = end

        # confusion_matrix = metrics.confusion_matrix(trues, preds, labels=None, sample_weight=None)
        acc = metrics.accuracy_score(trues, preds, normalize=True, sample_weight=None)
        # print("Confusion Matrix:\n{}".format(confusion_matrix), flush=True)
        print("Batch size: {}\t Acc.: {:.4f}".format(batch_size, acc), flush=True)
        batch_acc_list.append((batch_size, acc))
        print()
    return batch_acc_list

def evaluate_test_threshold(data_generator, data, label_num, thresholds, note):
    print("\n*******************Start to Zero-Shot predict on 【{}】 with thresholds*******************".format(
        note), flush=True)
    logits = []
    id2logit = {}
    id = 0
    for (x, _) in tqdm(data_generator):
        outputs = model.predict(x)
        for out in outputs:
            logit_pos = out[0].T
            logits.append(logit_pos)
            id2logit[id] = logit_pos
            id += 1

    sorted_logits = sorted(logits, reverse=True)
    for l in sorted_logits:
        print("{}".format(l))

    # Evaluate the results
    trues = [d[1] for d in data]
    preds = [0] * len(data)
    for id, logit in id2logit.items():
        label = 0
        while (label < label_num - 1):
            if (logit < thresholds[label]):
                break
            label += 1
        preds[id] = label

    confusion_matrix = metrics.confusion_matrix(trues, preds, labels=None, sample_weight=None)
    acc = metrics.accuracy_score(trues, preds, normalize=True, sample_weight=None)
    # print("Confusion Matrix:\n{}".format(confusion_matrix), flush=True)
    print("Acc.:\t{:.4f}".format(acc), flush=True)
    return acc

def divide_dict_batch(dict_a, batch_size):
    if (batch_size >= len(dict_a)):
        return [dict_a]
    else:
        sub_dicts = []
        list_a = [(k, v) for k, v in dict_a.items()]
        m = len(dict_a) // batch_size + 1
        for i in range(m):
            dict_b = {}
            start = i * batch_size
            if (i == m - 1):
                end = len(list_a)
            else:
                end = (i + 1) * batch_size
            for (k, v) in list_a[start:end]:
                dict_b[k] = v
            sub_dicts.append(dict_b)
        return sub_dicts

if __name__ == "__main__":

    time_start = time.time()
    # Load the hyper-parameters-----------------------------------------------------------
    maxlen = 128  # The max length 128 is used in our paper
    batch_size = 40  # Will not influence the results

    # Choose a model----------------------------------------------------------------------
    # Recommend to use 'uer-mixed-bert-base'
    # model_names = ['google-bert', 'google-bert-small', 'google-bert-zh',
    #                'hfl-bert-wwm', 'hfl-bert-wwm-ext',
    #                'uer-mixed-bert-tiny', 'uer-mixed-bert-small',
    #                'uer-mixed-bert-base', 'uer-mixed-bert-large']
    model_name = 'uer-mixed-bert-base'

    # Choose a dataset----------------------------------------------------------------------
    # dataset_names = ['cluewsc']
    dataset_name = 'cluewsc'

    # Load model and dataset class
    bert_model = Model(model_name=model_name)
    dataset = Datasets(dataset_name=dataset_name)

    # Choose a template [0, 1]--------------------------------------------------------
    pattern = dataset.patterns[1]

    # Prefix or Suffix-------------------------------------------------------------------
    is_pre = True

    # Load the dev set--------------------------------------------------------------------
    # -1 for all the samples
    dev_data = dataset.load_data(dataset.dev_path, sample_num=-1)
    dev_generator = data_generator(is_pre=is_pre,  pattern=pattern, data=dev_data, batch_size=batch_size)

    # Load the test set--------------------------------------------------------------------
    # -1 for all the samples
    test_data = dataset.load_data(dataset.test_path, sample_num=-1)
    test_generator = data_generator(is_pre=is_pre, pattern=pattern, data=test_data, batch_size=batch_size)

    # Build BERT model---------------------------------------------------------------------
    tokenizer = Tokenizer(bert_model.dict_path, do_lower_case=True)
    # Load BERET model with NSP head
    model = build_transformer_model(
        config_path=bert_model.config_path, checkpoint_path=bert_model.checkpoint_path, with_nsp=True,
    )

    # Zero-Shot predict and evaluate.-------------------------------------------------------
    # Predict on dev set and get the thresholds.
    _, thresholds = evaluate_dev(dev_generator, dev_data, len(dataset.labels), note="Dev Set")
    print("Thresholds of 【{}】 on dev set: {}".format(dataset_name, thresholds))

    # Predict by batched test set.
    evaluate_test_batch(test_generator, test_data, len(dataset.labels), note="Test Set")

    # Predict by thresholds.
    evaluate_test_threshold(test_generator, test_data, len(dataset.labels), thresholds=thresholds, note="Test Set")

    # Report the time cost.
    time_end = time.time()
    time_cost = time_end - time_start
    print("Time cost: {:.1f}s".format(time_cost))
