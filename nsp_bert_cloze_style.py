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

class data_generator(DataGenerator):
    """Data Generator"""

    def __init__(self, is_pre=True, is_soft_pos=False, *args, **kwargs):
        super(data_generator, self).__init__(*args, **kwargs)
        self.is_pre = is_pre
        self.is_soft_pos = is_soft_pos

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        batch_position_ids = []
        for is_end, (text, candi) in self.sample(random):
            text_1, text_2 = text.split("#idiom#")
            if (self.is_soft_pos):
                text = text.replace("#idiom#", "")
            if (self.is_pre):
                token_ids, segment_ids = tokenizer.encode(first_text=candi, second_text=text, maxlen=maxlen)
            else:
                token_ids, segment_ids = tokenizer.encode(first_text=text, second_text=candi, maxlen=maxlen)
            position_ids = []
            if (self.is_soft_pos):
                tokens_1 = tokenizer._tokenize(text_1)
                tokens_2 = tokenizer._tokenize(text_2)
                tokens_c = tokenizer._tokenize(candi)
                position_ids = [0]  # "[CSL]
                if (self.is_pre):
                    position_ids += [2 + len(tokens_1) + p_id for p_id in
                                     range(len(tokens_c))]  # "[CLS]" + TEXT1 + TEXT2 + "[SEP]" + CANDI
                    position_ids += [1]  # "[CLS]" + TEXT1 + TEXT2 + "[SEP]"
                    position_ids += [position_ids[-1] + p_id + 1 for p_id in range(len(tokens_1))]  # "[CLS]" + TEXT1
                    position_ids += [position_ids[-1] + len(tokens_c) + p_id + 1 for p_id in
                                     range(len(tokens_2))]  # "[CLS]" + TEXT1 +TEXT2
                    position_ids += [position_ids[-1] + 1]

                else:
                    position_ids += [position_ids[-1] + p_id + 1 for p_id in range(len(tokens_1))]  # "[CLS]" + TEXT1
                    position_ids += [position_ids[-1] + len(tokens_c) + p_id + 1 for p_id in
                                     range(len(tokens_2))]  # "[CLS]" + TEXT1 +TEXT2
                    position_ids += [position_ids[-1] + 1]  # "[CLS]" + TEXT1 + TEXT2 + "[SEP]"
                    position_ids += [1 + len(tokens_1) + p_id for p_id in
                                     range(len(tokens_c))]  # "[CLS]" + TEXT1 + TEXT2 + "[SEP]" + CANDI
                    position_ids += [len(tokens_1 + tokens_2 + tokens_c) + 2]
                    # position_ids += [position_ids[-1] + 1]

            source_ids, target_ids = token_ids[:], token_ids[:]
            # label_ids = tokenizer.encode(label)[0][1:-1]

            batch_token_ids.append(source_ids)
            batch_segment_ids.append(segment_ids)
            batch_position_ids.append(position_ids)

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_position_ids = sequence_padding(batch_position_ids)
                if (self.is_soft_pos):
                    yield [batch_token_ids, batch_segment_ids, batch_position_ids], None
                else:
                    yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids, batch_position_ids = [], [], []

def evaluate(data_generator_list, data, note=""):
    print("\n*******************Start to Zero-Shot predict on 【{}】*******************".format(note), flush=True)
    candidates_logits = [[] for _ in range(len(data[0][-1]))]
    for i in range(len(data_generator_list)):
        print("\nPattern{}".format(i), flush=True)
        data_generator = data_generator_list[i]
        counter = 0
        for (x, _) in tqdm(data_generator):
            outputs = model.predict(x)
            for out in outputs:
                logit_pos = out[0].T
                candidates_logits[i].append(logit_pos)
                counter += 1

    # Evaluate the results
    trues = [d[1] for d in data]
    preds = []
    for i in range(len(candidates_logits[0])):
        pred = numpy.argmax([logits[i] for logits in candidates_logits])
        preds.append(int(pred))

    confusion_matrix = metrics.confusion_matrix(trues, preds, labels=None, sample_weight=None)
    acc = metrics.accuracy_score(trues, preds, normalize=True, sample_weight=None)
    print("Confusion Matrix:\n{}".format(confusion_matrix), flush=True)
    print("Acc.:\t{:.4f}".format(acc), flush=True)
    return acc

def depart_choze(data):
    text_list = []
    candidates_list = [[] for _ in range(len(data[0][-1]))]
    for d in data:
        text = d[0]
        text_list.append(text)
        candidates = d[-1]
        for i, c in enumerate(candidates):
            candidates_list[i].append(c)
    return text_list, candidates_list

if __name__ == "__main__":

    # Load the hyper-parameters-----------------------------------------------------------
    maxlen = 128  # The max length 128 is used in our paper
    batch_size = 40  # Will not influence the results

    # Choose one of the models----------------------------------------------------------------------
    # Recommend to use 'uer-mixed-bert-base'
    # model_names = ['google-bert', 'hfl-bert-wwm', 'hfl-bert-wwm-ext',
    #                'uer-mixed-bert-tiny', 'uer-mixed-bert-small',
    #                'uer-mixed-bert-base', 'uer-mixed-bert-large']
    model_name = 'uer-mixed-bert-base'

    # Choose a dataset----------------------------------------------------------------------
    # dataset_names = ['chid']
    dataset_name = 'chid'

    # Load model and dataset class
    bert_model = Model(model_name=model_name)
    dataset = Datasets(dataset_name=dataset_name)

    # Prefix or Suffix-------------------------------------------------------------------
    is_pre = True
    # if using soft position or not
    is_soft_pos = True

    # Load the dev set--------------------------------------------------------------------
    # -1 for all the samples
    dev_data = dataset.load_data(dataset.dev_path, sample_num=-1, is_shuffle=True)
    text_list, candidates_list = depart_choze(dev_data)
    dev_generator_list = []
    for candidates in candidates_list:
        dev_generator_list.append(data_generator(is_pre=is_pre, is_soft_pos=is_soft_pos, data=zip(text_list, candidates),
                                                 batch_size=batch_size))

    # Load the test set--------------------------------------------------------------------
    # -1 for all the samples-
    test_data = dataset.load_data(dataset.test_path, sample_num=-1, is_shuffle=True)
    text_list, candidates_list = depart_choze(test_data)
    test_generator_list = []
    for candidates in candidates_list:
        test_generator_list.append(
            data_generator(is_pre=is_pre, is_soft_pos=is_soft_pos, data=zip(text_list, candidates),
                           batch_size=batch_size))

    # Build BERT model---------------------------------------------------------------------
    tokenizer = Tokenizer(bert_model.dict_path, do_lower_case=True)
    # Load BERET model with NSP head
    model = build_transformer_model(
        config_path=bert_model.config_path,
        checkpoint_path=bert_model.checkpoint_path,
        custom_position_ids=is_soft_pos,
        with_nsp=True,
    )

    # Zero-Shot predict and evaluate-------------------------------------------------------
    evaluate(dev_generator_list, dev_data, note="Dev Set")
    evaluate(test_generator_list, test_data, note="Test Set")
