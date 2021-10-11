#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge_sy"
# Date: 2021/7/6

import numpy as np
from tqdm import tqdm
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding, DataGenerator
from utils import *

# Mapping the labels to fixed-length span
duel_label_map = {'NIL_Event': '事件', 'NIL_Person': '人物', 'NIL_Work': '作品', 'NIL_Location': '区域',
                  'NIL_Time&Calendar': '时历', 'NIL_Brand': '品牌', 'NIL_Natural&Geography': '自然',
                  'NIL_Game': '游戏', 'NIL_Biological': '生物', 'NIL_Medicine': '药物', 'NIL_Food': '食物',
                  'NIL_Software': '软件', 'NIL_Vehicle': '车辆', 'NIL_Website': '网站', 'NIL_Disease&Symptom': '疾病',
                  'NIL_Organization': '组织', 'NIL_Awards': '奖项', 'NIL_Education': '教育', 'NIL_Culture': '文化',
                  'NIL_Constellation': '星座', 'NIL_Law&Regulation': '法律', 'NIL_VirtualThings': '虚拟',
                  'NIL_Diagnosis&Treatment': '诊断', 'NIL_Other': '其他'}

class data_generator(DataGenerator):
    """Data Generator"""
    def __init__(self, is_pre=True,  is_two_stage=False, *args, **kwargs):
        super(data_generator, self).__init__(*args, **kwargs)
        self.is_pre = is_pre
        self.is_two_stage = is_two_stage

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_mask_ids, batch_labels = [], [], [], []
        for is_end, (
                text_id, mention_id, mention, offset, kind, label, type_know_id, text, type_know_text) in self.sample(
            random):
            if(self.is_two_stage):
                text = "{}上文中{}是指".format(text, mention)
            if (self.is_pre):
                token_ids, segment_ids = tokenizer.encode(first_text=type_know_text, second_text=text, maxlen=maxlen)
            else:
                token_ids, segment_ids = tokenizer.encode(first_text=text, second_text=type_know_text, maxlen=maxlen)

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def get_el_data(dev_data_original, dataset, mention2ids, id2data, data_kind='both', is_with_other=True):
    """
    According to "dev.json" and knowledge base "kb.json" to make test dataset for BERT-NSP
    Format：(text_id, mention_id, 'type'/'subject', label, type_id/know_id, text, type_text/k_text)
    """
    type_list = dataset.type_list
    type_en2zh = dataset.type_en2zh
    dev_data = []
    line_data_type = []
    line_data_entity = []
    line_data_error = []
    line_data_other = []
    true_results = []
    error_count = 0
    other_count = 0
    sum_entity_num = 0
    for line in dev_data_original:
        text_id = line["text_id"]
        mention_data = line["mention_data"]
        text = line["text"]
        for m_id, m_d in enumerate(mention_data):
            kb_id = m_d["kb_id"]
            mention = m_d["mention"]
            offset = int(m_d["offset"])

            # The mention can be linked to Knowledge Base
            if (mention in mention2ids and kb_id in id2data):
                ids = mention2ids[mention]
                sum_entity_num += len(ids)
                if (kb_id in ids):
                    if(data_kind=='only-entity' or data_kind=='both'):
                        line_data_entity.append((text_id, m_id, mention, offset, kb_id))
                        for j, id in enumerate(ids):
                            k_text = ""
                            k_text_extra = ""
                            data = id2data[id]
                            label = 1 if id == kb_id else 0
                            for triple in data:
                                pred = triple["predicate"]
                                obj = triple["object"]
                                if (pred == "摘要"):
                                    obj = obj.replace(mention, "")
                                    k_text = "这个词是指{}".format(obj)
                                else:
                                    k_text_extra += "这个词的{}是{}；".format(pred, obj)
                            if (k_text == ""):
                                k_text = k_text_extra
                            dev_data.append((text_id, m_id, mention, offset, 'subject', label, id, text, k_text))
                            if (label == 1):
                                true_results.append((text_id, m_id, id))
                else:
                    # The mention can be found in Knowledge Base, but don't belong to anyone of thems.
                    print(line)
                    line_data_error.append((text_id, m_id, mention, offset, kb_id))
                    error_count += 1
            else:
                # The mention is not in Knowledge Base
                line_data_type.append((text_id, m_id, mention, offset, kb_id))
                if(data_kind=='only-type' or data_kind=='both'):

                    if ('_' in kb_id):
                        true_type = kb_id.split('_')[-1]
                        type_index = type_list.index(true_type)
                    else:
                        type_index = -1
                    if(is_with_other==False and kb_id=="NIL_Other"):
                        other_count += 1
                        line_data_other.append((text_id, m_id, mention, offset, kb_id))
                        continue
                    else:
                        for j, type in enumerate(duel_label_map.values()):
                            label = 1 if j == type_index else 0
                            type_text = "这个词是指一类{}".format(type)
                            dev_data.append((text_id, m_id, mention, offset, 'type', label, j, text, type_text))
                            if (label == 1):
                                true_results.append((text_id, m_id, j))

    print("Entity Linking Samples number: {}".format(len(line_data_entity)))
    print("Entity Typing Samples number: {}".format(len(line_data_type)))
    print("Bad Sample number: {}".format(len(line_data_error)))
    print("The type Other number：{}".format(len(line_data_other)))
    print("Prediction part：{}".format(data_kind))
    return dev_data, true_results


def evaluate(data_generator, data, true_results, note=""):

    print("\n*******************Start to Zero-Shot predict on 【{}】*******************".format(note), flush=True)
    counter = 0
    logits = []
    logit_data = []
    for (x, _) in tqdm(data_generator):
        outputs = model.predict(x)
        for out in outputs:
            logit_pos = out[0].T
            logits.append(logit_pos)
            logit_data.append((logit_pos, data[counter]))
            counter += 1

    # Evaluate the results
    trues = [d[1] for d in data]
    preds = []
    text_id, mention_id, start_i = -1, -1, 0
    pred_results = []
    logits = []
    for i, (logit, d) in enumerate(logit_data):
        new_text_id = d[0]
        new_mention_id = d[1]
        if (text_id == -1 or (new_text_id == text_id and new_mention_id == mention_id)):
            logits.append(logit)
            text_id, mention_id = new_text_id, new_mention_id
        else:
            # Deal with the last mention
            argmax = np.argmax(logits)
            result_id = data[start_i + argmax][6]
            pred_results.append((text_id, mention_id, result_id))
            # Record new results
            logits = [logit]
            text_id, mention_id = new_text_id, new_mention_id
            start_i = i
        if (i == len(logit_data) - 1):
            # Deal with the last mention
            argmax = np.argmax(logits)
            result_id = data[start_i + argmax][6]
            pred_results.append((text_id, mention_id, result_id))

    # Calculate Acc.
    acc_count = 0
    assert len(pred_results) == len(true_results)
    for pred, true in zip(pred_results, true_results):
        pred_id = pred[-1]
        true_id = true[-1]
        if (pred_id == true_id):
            acc_count += 1
    acc = acc_count / len(pred_results)
    print("Acc.:\t{:.4f}".format(acc), flush=True)
    return acc

if __name__ == "__main__":

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
    # dataset_names = ['duel2.0']
    dataset_name = 'duel2.0'

    # Load model and dataset class
    bert_model = Model(model_name=model_name)
    dataset = Datasets(dataset_name=dataset_name)

    # Entity Linking or Entity Typing or both------------------------------------------------
    # 'only-entity' for Entity Linking, 'only-type' for Entity Typing
    # data_kinds = ['only-entity', 'only-type', 'both']
    data_kind = 'only-type'

    # If with 'Other' type-------------------------------------------------------------------
    is_with_other = True

    # Two-stage prompt-----------------------------------------------------------------------
    is_two_stage = True

    # Read the Knowledge Base-----------------------------------------------------------------
    kb_list, mention2ids, id2data, id2type = dataset.load_kb(dataset.kb_path)

    # Load the dev set------------------------------------------------------------------------
    # -1 for all the samples
    dev_data_original = dataset.load_data(dataset.dev_path, sample_num=-1)
    dev_data, true_results = get_el_data(dev_data_original, dataset, mention2ids, id2data, data_kind=data_kind, is_with_other=is_with_other)
    dev_generator = data_generator(is_pre=False, is_two_stage=is_two_stage,data=dev_data, batch_size=batch_size)

    # Build BERT model------------------------------------------------------------------------
    tokenizer = Tokenizer(bert_model.dict_path, do_lower_case=True)
    # Load BERET model with NSP head
    model = build_transformer_model(
        config_path=bert_model.config_path, checkpoint_path=bert_model.checkpoint_path, with_nsp=True,
    )

    # Zero-Shot predict and evaluate----------------------------------------------------------
    acc = evaluate(dev_generator, dev_data, true_results, note='Dev set')
    # evaluate(test_generator, test_data)
