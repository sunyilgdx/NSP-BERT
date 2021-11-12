#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge_sy"
# Date: 2021/11/5

MODEL_NAME = {
    # FewCLUE
    'eprstmt': 'uer-mixed-bert-base',
    'tnews': 'uer-mixed-bert-base',
    'csldcp': 'uer-mixed-bert-base',
    'iflytek': 'uer-mixed-bert-base',
    # English datasets in KPT
    'AGNews': 'google-bert',
    'DBPedia': 'google-bert',
    'IMDB': 'google-bert',
    'Amazon': 'google-bert',
    # GLUE
    'CoLA': 'google-bert-cased',
    'SST-2': 'google-bert-cased',
    'MNLI': 'google-bert-cased-wwm-large',
    'MNLI-mm': 'google-bert-cased-wwm-large',
    'QNLI': 'google-bert-cased-wwm-large',
    'RTE': 'google-bert-cased-wwm-large',
    'MRPC': 'google-bert-cased-wwm-large',
    'QQP': 'google-bert-cased-wwm-large',
    'STS-B': 'google-bert-cased-wwm-large',
    'WNLI': 'google-bert-cased-wwm-large',
    # Others in LM-BFF
    'SST-5': 'google-bert-cased',
    'MR': 'google-bert-cased',
    'CR': 'google-bert-cased',
    'MPQA': 'google-bert-cased',
    'Subj': 'google-bert-cased',
    'TREC': 'google-bert-cased',
    'SNLI': 'google-bert-cased-wwm-large',

}

IS_PRE = {
    # FewCLUE
    'eprstmt': True,
    'tnews': True,
    'csldcp': True,
    'iflytek': True,
    'bustm': True,
    'ocnli': True,
    'csl': False,
    # English datasets in KP
    'AGNews': True,
    'DBPedia': False,
    'IMDB': True,
    'Amazon': True,
    # GLUE
    'CoLA': True,
    'SST-2': False,
    'MNLI': False,
    'MNLI-mm': False,
    'QNLI': False,
    'RTE': True,
    'MRPC': False,
    'QQP': False,
    'STS-B': True,
    'WNLI': False,
    'SNLI': False,
    # Others in LM-BFF
    'SST-5': False,
    'MR': True,
    'CR': True,
    'MPQA': True,
    'Subj': True,
    'TREC': True,
}

PATTERN_INDEX = {
    # FewCLUE
    'eprstmt': -1,
    'tnews': -1,
    'csldcp': -1,
    'iflytek': -1,
    # English datasets in KP
    'AGNews': -1,
    'DBPedia': 0,
    'IMDB': -1,
    'Amazon': -1,
    # GLUE
    'CoLA': -1,
    'SST-2': -1,
    # Others in LM-BFF
    'SST-5': -1,
    'MR': -1,
    'CR': -1,
    'MPQA': -1,
    'Subj': -1,
    'TREC': -1,
}

# Sample number per class
K_SHOT = {
    # FewCLUE
    'eprstmt': 16,
    'tnews': 16,
    'csldcp': 8,
    'iflytek': 8,
    'bustm': -1,
    'ocnli': -1,
    'csl': -1,
    # English datasets in KP
    'AGNews': 16,
    'DBPedia': 16,
    'IMDB': 16,
    'Amazon': 16,
    # GLUE
    'CoLA': 16,
    'SST-2': 16,
    'MNLI': 16,
    'MNLI-mm': 16,
    'QNLI': 16,
    'RTE': 16,
    'MRPC': 16,
    'QQP': 16,
    'STS-B': 96, # Regression [0, 5]
    'WNLI': 16,
    # Others in LM-BFF
    'SST-5': 80,
    'MR': 16,
    'CR': 16,
    'MPQA': 16,
    'Subj': 16,
    'TREC': 16,
    'SNLI': 16,
}
