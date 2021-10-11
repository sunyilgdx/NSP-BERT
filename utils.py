#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge_sy"
# Date: 2021/6/30
import csv
import json
import random

class Datasets():
    def __init__(self, dataset_name=""):
        self.dataset_name = dataset_name
        self.patterns = []
        self.train_path, self.dev_path, self.test_path = "", "", ""

        if (dataset_name == 'eprstmt'):
            self.train_path = r"./datasets/clue_datasets/eprstmt/train_few_all.json"
            self.dev_path = r"./datasets/clue_datasets/eprstmt/dev_few_all.json"
            self.test_path = r"./datasets/clue_datasets/eprstmt/test_public.json"
            self.label_texts = ["Positive", "Negative"]
            self.text2id = {"Positive": 0, "Negative": 1}
            self.patterns = [['好评', '差评'], ['东西不错', '东西很差'], ['这次买的东西很好', '这次买的东西很差']]

        elif (dataset_name == "tnews"):
            self.train_path = r"./datasets/few_clue/tnews_new/train_few_all.json"
            self.dev_path = r"./datasets/few_clue/tnews_new/dev_few_all.json"
            self.test_path = r"./datasets/few_clue/tnews_new/test_public.json"
            self.label_texts = ["文化", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技", "军事", "旅游", "国际",
                                "证券", "农业", "电竞", "民生"]
            self.templates = ["[label]", "[label]新闻", "这是一则[label]新闻"]
            self.patterns = [[template.replace('[label]', label) for label in self.label_texts] for template in
                             self.templates]

        elif (dataset_name == "csldcp"):
            self.train_path = r"./datasets/clue_datasets/csldcp/train_few_all.json"
            self.dev_path = r"./datasets/clue_datasets/csldcp/dev_few_all.json"
            self.test_path = r"./datasets/clue_datasets/csldcp/test_public.json"
            self.label_path = r"./datasets/clue_datasets/csldcp/labels_all.txt"
            self.label_texts, self.text2id = read_labels(self.label_path)
            self.templates = ["[label]", "[label]类论文", "这是一篇[label]类论文"]
            self.patterns = [[template.replace('[label]', label) for label in self.label_texts] for template in
                             self.templates]

        elif (dataset_name == 'iflytek'):
            self.train_path = r"./datasets/clue_datasets/iflytek/train_few_all.json"
            self.dev_path = r"./datasets/clue_datasets/iflytek/dev_few_all.json"
            self.test_path = r"./datasets/clue_datasets/iflytek/test_public.json"
            self.label_text2label_id = {  # 中文标签对应的ID
                "打车": 0, "美颜": 100, "影像剪辑": 101, "摄影修图": 102,
                "相机": 103, "绘画": 104, "二手": 105, "电商": 106,
                "团购": 107, "外卖": 108, "电影票务": 109, "社区服务": 10,
                "社区超市": 110, "购物咨询": 111, "笔记": 112, "办公": 113,
                "日程管理": 114, "女性": 115, "经营": 116, "收款": 117,
                "其他": 118, "薅羊毛": 11, "魔幻": 12, "仙侠": 13,
                "卡牌": 14, "飞行空战": 15, "射击游戏": 16, "休闲益智": 17,
                "动作类": 18, "体育竞技": 19, "地图导航": 1, "棋牌中心": 20,
                "经营养成": 21, "策略": 22, "MOBA": 23, "辅助工具": 24,
                "约会社交": 25, "即时通讯": 26, "工作社交": 27, "论坛圈子": 28,
                "婚恋社交": 29, "免费WIFI": 2, "情侣社交": 30, "社交工具": 31,
                "生活社交": 32, "微博博客": 33, "新闻": 34, "漫画": 35,
                "小说": 36, "技术": 37, "教辅": 38, "问答交流": 39,
                "租车": 3, "搞笑": 40, "杂志": 41, "百科": 42,
                "影视娱乐": 43, "求职": 44, "兼职": 45, "视频": 46,
                "短视频": 47, "音乐": 48, "直播": 49, "同城服务": 4,
                "电台": 50, "K歌": 51, "成人": 52, "中小学": 53,
                "职考": 54, "公务员": 55, "英语": 56, "视频教育": 57,
                "高等教育": 58, "成人教育": 59, "快递物流": 5, "艺术": 60,
                "语言(非英语)": 61, "旅游资讯": 62, "综合预定": 63, "民航": 64,
                "铁路": 65, "酒店": 66, "行程管理": 67,
                "民宿短租": 68, "出国": 69, "婚庆": 6, "工具": 70,
                "亲子儿童": 71, "母婴": 72, "驾校": 73, "违章": 74,
                "汽车咨询": 75, "汽车交易": 76, "日常养车": 77, "行车辅助": 78,
                "租房": 79, "家政": 7, "买房": 80, "装修家居": 81,
                "电子产品": 82, "问诊挂号": 83, "养生保健": 84, "医疗服务": 85,
                "减肥瘦身": 86, "美妆美业": 87, "菜谱": 88, "餐饮店": 89,
                "公共交通": 8, "体育咨讯": 90, "运动健身": 91, "支付": 92,
                "保险": 93, "股票": 94, "借贷": 95, "理财": 96,
                "彩票": 97, "记账": 98, "银行": 99, "政务": 9,
            }
            self.label_texts = ['打车', '地图导航', '免费WIFI', '租车', '同城服务', '快递物流', '婚庆', '家政', '公共交通', '政务', '社区服务', '薅羊毛',
                                '魔幻', '仙侠', '卡牌', '飞行空战', '射击游戏', '休闲益智', '动作类', '体育竞技', '棋牌中心', '经营养成', '策略', 'MOBA',
                                '辅助工具', '约会社交', '即时通讯', '工作社交', '论坛圈子', '婚恋社交', '情侣社交', '社交工具', '生活社交', '微博博客', '新闻',
                                '漫画', '小说', '技术', '教辅', '问答交流', '搞笑', '杂志', '百科', '影视娱乐', '求职', '兼职', '视频', '短视频', '音乐',
                                '直播', '电台', 'K歌', '成人', '中小学', '职考', '公务员', '英语', '视频教育', '高等教育', '成人教育', '艺术',
                                '语言(非英语)', '旅游资讯', '综合预定', '民航', '铁路', '酒店', '行程管理', '民宿短租', '出国', '工具', '亲子儿童', '母婴',
                                '驾校', '违章', '汽车咨询', '汽车交易', '日常养车', '行车辅助', '租房', '买房', '装修家居', '电子产品', '问诊挂号', '养生保健',
                                '医疗服务', '减肥瘦身', '美妆美业', '菜谱', '餐饮店', '体育咨讯', '运动健身', '支付', '保险', '股票', '借贷', '理财', '彩票',
                                '记账', '银行', '美颜', '影像剪辑', '摄影修图', '相机', '绘画', '二手', '电商', '团购', '外卖', '电影票务', '社区超市',
                                '购物咨询', '笔记', '办公', '日程管理', '女性', '经营', '收款', '其他']
            self.templates = ["[label]", "[label]类软件", "这是一款[label]类软件"]
            self.patterns = [[template.replace('[label]', label) for label in self.label_texts] for template in
                             self.templates]

        elif (dataset_name == "ocnli"):
            self.train_path = r"./datasets/clue_datasets/ocnli/train_few_all.json"
            self.dev_path = r"./datasets/clue_datasets/ocnli/dev_few_all.json"
            self.test_path = r"./datasets/clue_datasets/ocnli/test_public.json"
            self.labels = [0, 1, 2]
            self.label_texts = ["entailment", "contradiction", "neutral"]
            self.label_text2label_id = {"entailment": 2, "contradiction": 0, "neutral": 1}

        elif (dataset_name == "bustm"):
            self.train_path = r"./datasets/clue_datasets/bustm/train_few_all.json"
            self.dev_path = r"./datasets/clue_datasets/bustm/dev_few_all.json"
            self.test_path = r"./datasets/clue_datasets/bustm/test_public.json"
            self.labels = [0, 1]

        elif (dataset_name == "chid"):
            self.train_path = r"./datasets/clue_datasets/chid/train_few_all.json"
            self.dev_path = r"./datasets/clue_datasets/chid/dev_few_all.json"
            self.test_path = r"./datasets/clue_datasets/chid/test_public.json"

        elif (dataset_name == "csl"):
            self.train_path = r"./datasets/clue_datasets/csl/train_few_all.json"
            self.dev_path = r"./datasets/clue_datasets/csl/dev_few_all.json"
            self.test_path = r"./datasets/clue_datasets/csl/test_public.json"
            self.labels = [0, 1]

        elif (dataset_name == "cluewsc"):
            self.train_path = r"./datasets/clue_datasets/cluewsc/train_few_all.json"
            self.dev_path = r"./datasets/clue_datasets/cluewsc/dev_few_all.json"
            self.test_path = r"./datasets/clue_datasets/cluewsc/test_public.json"
            self.labels = [0, 1]
            self.text2id = {"true": 1, "false": 0}
            self.patterns = ['其中', '上文中']

        elif (dataset_name == "duel2.0"):
            self.train_path = r"./datasets/DuEL 2.0/train.json"
            self.dev_path = r"./datasets/DuEL 2.0/dev.json"
            self.test_path = r"./datasets/DuEL 2.0/test.json"
            self.kb_path = r"./datasets/DuEL 2.0/kb.json"
            self.type_en2zh = {'Event': '事件活动', 'Person': '人物', 'Work': '作品', 'Location': '区域场所',
                               'Time&Calendar': '时间历法', 'Brand': '品牌', 'Natural&Geography': '自然地理',
                               'Game': '游戏', 'Biological': '生物', 'Medicine': '药物', 'Food': '食物',
                               'Software': '软件', 'Vehicle': '车辆', 'Website': '网站平台', 'Disease&Symptom': '疾病症状',
                               'Organization': '组织机构', 'Awards': '奖项', 'Education': '教育', 'Culture': '文化',
                               'Constellation': '星座', 'Law&Regulation': '法律法规', 'VirtualThings': '虚拟事物',
                               'Diagnosis&Treatment': '诊断治疗方法', 'Other': '其他'}
            self.type_list = ['Event', 'Person', 'Work', 'Location', 'Time&Calendar', 'Brand', 'Natural&Geography',
                              'Game', 'Biological', 'Medicine', 'Food', 'Software', 'Vehicle', 'Website',
                              'Disease&Symptom',
                              'Organization', 'Awards', 'Education', 'Culture', 'Constellation', 'Law&Regulation',
                              'VirtualThings',
                              'Diagnosis&Treatment', 'Other']

        elif (dataset_name == "AGNews"):
            self.train_path = r"./datasets/enEval/agnews/train.csv"
            self.dev_path = r"./datasets//enEval/agnews/test.csv"
            self.test_path = r"./datasets//enEval/agnews/test.csv"
            self.label_texts = ["political", "sports", "business", "technology"]
            self.templates = ["[label]", "This is a [label] news", "The above news is about [label]"]
            self.patterns = [[template.replace('[label]', label) for label in self.label_texts] for template in
                             self.templates]

        elif (dataset_name == "DBPedia"):
            self.train_path = r"./datasets/enEval/dbpedia/train.txt"
            self.dev_path = r"./datasets//enEval/dbpedia/test.txt"
            self.test_path = r"./datasets//enEval/dbpedia/test.txt"
            self.label_texts = ["company", "school university", "artist", "athlete", "politics", "transportation",
                                "building",
                                "river mountain lake", "village", "animal", "plant tree", "album", "film",
                                "book publication"]
            self.templates = ["[label]", "This is about [label]", "It's a [label]"]
            self.patterns = [[template.replace('[label]', label) for label in self.label_texts] for template in
                             self.templates]

        elif (dataset_name == "IMDB"):
            self.train_path = r"./datasets/enEval/imdb/train.txt"
            self.dev_path = r"./datasets/enEval/imdb/test.txt"
            self.test_path = r"./datasets/enEval/imdb/test.txt"
            self.label_texts = ['bad', 'good']
            self.templates = ["It is [label]", "This movie is [label]",
                              "After watching this movie, I think it's [label]"]
            self.patterns = [[template.replace('[label]', label) for label in self.label_texts] for template in
                             self.templates]

        elif (dataset_name == "Amazon"):
            self.train_path = r"./datasets/enEval/amazon/train.txt"
            self.dev_path = r"./datasets/enEval/amazon/test.txt"
            self.test_path = r"./datasets/enEval/amazon/test.txt"
            self.label_texts = ['bad', 'good']
            self.templates = ["It is [label]", "All in all, it is [label]",
                              "I think it is [label]"]
            self.patterns = [[template.replace('[label]', label) for label in self.label_texts] for template in
                             self.templates]

    def load_data(self, filename, sample_num=-1, is_train=False, is_shuffle=False):
        D = []

        if (self.dataset_name == "eprstmt"):
            text2id = self.text2id
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    content = json.loads(l)['sentence']
                    label_text = json.loads(l)['label']
                    label_id = text2id[label_text]
                    D.append((content, int(label_id)))

        elif (self.dataset_name == "tnews"):
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    text = json.loads(l)['text']
                    label = json.loads(l)['label']
                    D.append((text, int(label)))

        elif (self.dataset_name == "csldcp"):
            text2id = self.text2id
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    content = json.loads(l)['content']
                    label_text = json.loads(l)['label']
                    label_id = text2id[label_text]
                    D.append((content, int(label_id)))

        elif (self.dataset_name == "iflytek"):
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    text = json.loads(l)['sentence']
                    label = json.loads(l)['label']
                    D.append((text, int(label)))

        elif (self.dataset_name == "ocnli"):
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    sentence1 = json.loads(l)['sentence1']
                    sentence2 = json.loads(l)['sentence2']
                    label_text = json.loads(l)['label']
                    label = int(self.label_text2label_id[label_text])
                    text = "{}[SEP]{}".format(sentence1, sentence2)
                    D.append((text, int(label)))

        elif (self.dataset_name == "bustm"):
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    sentence1 = json.loads(l)['sentence1']
                    sentence2 = json.loads(l)['sentence2']
                    label = json.loads(l)['label']
                    text = "{}[SEP]{}".format(sentence1, sentence2)
                    D.append((text, int(label)))

        elif (self.dataset_name == "chid"):
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    content = json.loads(l)['content']
                    candidates = json.loads(l)['candidates']
                    label = json.loads(l)['answer']
                    D.append((content, int(label), candidates))

        elif (self.dataset_name == "csl"):
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    content = json.loads(l)['abst']
                    keywords = json.loads(l)['keyword']
                    label = json.loads(l)['label']
                    D.append((content + "[SEP]" + ",".join(keywords), int(label)))

        elif (self.dataset_name == "cluewsc"):
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    target = json.loads(l)['target']
                    span1_text = target['span1_text']
                    span2_text = target['span2_text']
                    span1_index = target['span1_index']
                    span2_index = target['span2_index']
                    text = json.loads(l)['text']
                    label = json.loads(l)['label']
                    label = self.text2id[label]
                    D.append((text, int(label), span1_text, span2_text, span1_index, span2_index))

        elif (self.dataset_name == "AGNews"):
            with open(filename, encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=',')
                for idx, row in enumerate(reader):
                    label, headline, body = row
                    text_a = headline.replace('\\', ' ')
                    text_b = body.replace('\\', ' ')
                    D.append((text_a + ". " + text_b, int(label) - 1))
                    # D.append((text_b, int(label) - 1))

        elif (self.dataset_name == "DBPedia"):
            label_filename = ""
            if ('test' in filename):
                label_filename = filename.replace('test', 'test_labels')
            if ('train' in filename):
                label_filename = filename.replace('train', 'train_labels')
            lines = []
            entities = []
            with open(filename, encoding='utf-8') as f:
                for line in f.readlines():
                    lines.append(line)
                    entity = line.split('.')[0]
                    entities.append(entity)
            labels = []
            with open(label_filename, encoding='utf-8') as label_f:
                for label in label_f.readlines():
                    labels.append(label)
            for line, entity, label in zip(lines, entities, labels):
                text = "{} The {} is a".format(line, entity)
                D.append((text, int(label)))
            print("We recommend using template0 with the suffix.")

        elif (self.dataset_name == "IMDB"):
            label_filename = ""
            if ('test' in filename):
                label_filename = filename.replace('test', 'test_labels')
            if ('train' in filename):
                label_filename = filename.replace('train', 'train_labels')
            lines = []
            with open(filename, encoding='utf-8') as f:
                for line in f.readlines():
                    lines.append(line)
            labels = []
            with open(label_filename, encoding='utf-8') as label_f:
                for label in label_f.readlines():
                    labels.append(label)
            for line, label in zip(lines, labels):
                D.append((line, int(label)))

        elif (self.dataset_name == "Amazon"):
            label_filename = ""
            if ('test' in filename):
                label_filename = filename.replace('test', 'test_labels')
            if ('train' in filename):
                label_filename = filename.replace('train', 'train_labels')
            lines = []
            with open(filename, encoding='utf-8') as f:
                for line in f.readlines():
                    lines.append(line)
            labels = []
            with open(label_filename, encoding='utf-8') as label_f:
                for label in label_f.readlines():
                    labels.append(label)
            for line, label in zip(lines, labels):
                D.append((line, int(label)))

        elif (self.dataset_name == "duel2.0"):
            with open(filename, encoding='utf-8')as f:
                for l in f:
                    D.append(json.loads(l))

        # Shuffle the dataset.
        if (is_shuffle):
            random.seed(1)
            random.shuffle(D)

        # Set the number of samples.
        if (sample_num == -1):
            # -1 for all the samples
            return D
        else:
            return D[:sample_num + 1]

    # Load the Knowledge Base for DuEL2.0.
    def load_kb(self, filename):
        kb_list = []
        mention2id = {}
        id2data = {}
        id2type = {}
        with open(filename, "r", encoding='utf-8') as kb_file:
            for line in kb_file.readlines():
                k = json.loads(line)
                kb_list.append(k)
                subject_id = k["subject_id"]
                alias = k["alias"]
                data = k["data"]
                type = k["type"]
                id2data[subject_id] = data
                id2type[subject_id] = type
                subject = k["subject"]
                if (subject not in alias):
                    alias.append(subject)

                for alia in alias:
                    if (alia not in mention2id):
                        mention2id[alia] = set()
                        mention2id[alia].add(subject_id)
                    else:
                        mention2id[alia].add(subject_id)
        return kb_list, mention2id, id2data, id2type


class Model():

    def __init__(self, model_name=""):
        self.model_name = model_name
        self.config_path, self.checkpoint_path, self.dict_path = "", "", ""

        if (model_name == 'google-bert'):
            self.config_path = './models/uncased_L-12_H-768_A-12/bert_config.json'
            self.checkpoint_path = './models/uncased_L-12_H-768_A-12/bert_model.ckpt'
            self.dict_path = './models/uncased_L-12_H-768_A-12/vocab.txt'

        elif (model_name == 'google-bert-small'):
            self.config_path = './models/uncased_L-8_H-512_A-8/bert_config.json'
            self.checkpoint_path = './models/uncased_L-8_H-512_A-8/bert_model.ckpt'
            self.dict_path = './models/uncased_L-8_H-512_A-8/vocab.txt'

        if (model_name == 'google-bert-wwm-large'):
            self.config_path = './models/wwm_uncased_L-24_H-1024_A-16/bert_config.json'
            self.checkpoint_path = './models/wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt'
            self.dict_path = './models/wwm_uncased_L-24_H-1024_A-16/vocab.txt'

        elif (model_name == 'google-bert-zh'):
            self.config_path = './models/chinese_L-12_H-768_A-12/bert_config.json'
            self.checkpoint_path = './models/chinese_L-12_H-768_A-12/bert_model.ckpt'
            self.dict_path = './models/chinese_L-12_H-768_A-12/vocab.txt'

        elif (model_name == 'hfl-bert-wwm'):
            self.config_path = './models/chinese_wwm_L-12_H-768_A-12/bert_config.json'
            self.checkpoint_path = './models/chinese_wwm_L-12_H-768_A-12/bert_model.ckpt'
            self.dict_path = './models/chinese_wwm_L-12_H-768_A-12/vocab.txt'

        elif (model_name == 'hfl-bert-wwm-ext'):
            self.config_path = './models/chinese_wwm_ext_L-12_H-768_A-12/bert_config.json'
            self.checkpoint_path = './models/chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
            self.dict_path = './models/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt'

        elif (model_name == "uer-mixed-bert-tiny"):
            self.config_path = './models/uer_mixed_corpus_bert_tiny/bert_config.json'
            self.checkpoint_path = './models/uer_mixed_corpus_bert_tiny/bert_model.ckpt'
            self.dict_path = './models/uer_mixed_corpus_bert_tiny/vocab.txt'

        elif (model_name == "uer-mixed-bert-small"):
            self.config_path = './models/uer_mixed_corpus_bert_small/bert_config.json'
            self.checkpoint_path = './models/uer_mixed_corpus_bert_small/bert_model.ckpt'
            self.dict_path = './models/uer_mixed_corpus_bert_small/vocab.txt'

        elif (model_name == "uer-mixed-bert-base"):
            self.config_path = './models/uer_mixed_corpus_bert_base/bert_config.json'
            self.checkpoint_path = './models/uer_mixed_corpus_bert_base/bert_model.ckpt'
            self.dict_path = './models/uer_mixed_corpus_bert_base/vocab.txt'

        elif (model_name == "uer-mixed-bert-large"):
            self.config_path = './models/uer_mixed_corpus_bert_large/bert_config.json'
            self.checkpoint_path = './models/uer_mixed_corpus_bert_large/bert_model.ckpt'
            self.dict_path = './models/uer_mixed_corpus_bert_large/vocab.txt'


def read_labels(label_file_path):
    labels_text = []
    text2id = {}
    with open(label_file_path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f.readlines()):
            label = line.strip('\n')
            labels_text.append(label)
            text2id[label] = index
    return labels_text, text2id

# if __name__ == "__main__":
#     print()
