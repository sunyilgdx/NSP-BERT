#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge_sy"
# Date: 2021/6/30
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
            self.patterns = [['好评', '差评'], ['东西不错', '东西很差'],  ['这次买的东西很好', '这次买的东西很差']]

        elif (dataset_name == "tnews"):
            self.patterns = [["文化", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技", "军事", "旅游", "国际", "证券", "农业", "电竞", "民生"],
                             ["文化新闻", "娱乐新闻", "体育新闻", "财经新闻", "房产新闻", "汽车新闻", "教育新闻", "科技新闻", "军事新闻", "旅游新闻", "国际新闻",
                              "证券新闻", "农业新闻", "电竞新闻", "民生新闻"],
                             ["这是一则文化新闻", "这是一则娱乐新闻", "这是一则体育新闻", "这是一则财经新闻", "这是一则房产新闻", "这是一则汽车新闻", "这是一则教育新闻",
                              "这是一则科技新闻", "这是一则军事新闻", "这是一则旅游新闻", "这是一则国际新闻", "这是一则证券新闻", "这是一则农业新闻", "这是一则电竞新闻",
                              "这是一则民间民生故事新闻"]]
            self.train_path = r"./datasets/few_clue/tnews_new/train_few_all.json"
            self.dev_path = r"./datasets/few_clue/tnews_new/dev_few_all.json"
            self.test_path = r"./datasets/few_clue/tnews_new/test_public.json"
            self.label_pair = {"文化": ["娱乐", "教育", "旅游", "民生"], "娱乐": ["文化", "旅游", "电竞"], "体育": ["娱乐", "国际", "电竞"],
                               "财经": ["房产", "科技", "国际", "民生"], "房产": ["财经", "科技", "旅游"], "汽车": ["财经", "科技", "旅游"],
                               "教育": ["文化", "科技", "国际"], "科技": ["财经", "汽车", "电竞"], "军事": ["文化", "科技", "国际"],
                               "旅游": ["文化", "国际", "农业"], "国际": ["财经", "军事", "旅游", "民生"], "证券": ["财经", "军事", "国际"],
                               "农业": ["文化", "财经", "旅游"], "电竞": ["娱乐", "体育", "科技"], "故事": ["文化", "娱乐", "农业"],
                               }

        elif (dataset_name == "csldcp"):
            self.train_path = r"./datasets/clue_datasets/csldcp/train_few_all.json"
            self.dev_path = r"./datasets/clue_datasets/csldcp/dev_few_all.json"
            self.test_path = r"./datasets/clue_datasets/csldcp/test_public.json"
            self.label_path = r"./datasets/clue_datasets/csldcp/labels_all.txt"
            self.label_texts, self.text2id = read_labels(self.label_path)
            self.patterns = [
                ['材料科学与工程', '作物学', '口腔医学', '药学', '教育学', '水利工程', '理论经济学', '食品科学与工程', '畜牧学/兽医学', '体育学', '核科学与技术', '力学',
                 '园艺学', '水产', '法学', '地质学/地质资源与地质工程', '石油与天然气工程', '农林经济管理', '信息与通信工程', '图书馆、情报与档案管理', '政治学', '电气工程',
                 '海洋科学',
                 '民族学', '航空宇航科学与技术', '化学/化学工程与技术', '哲学', '公共卫生与预防医学', '艺术学', '农业工程', '船舶与海洋工程', '计算机科学与技术', '冶金工程',
                 '交通运输工程', '动力工程及工程热物理', '纺织科学与工程', '建筑学', '环境科学与工程', '公共管理', '数学', '物理学', '林学/林业工程', '心理学', '历史学',
                 '工商管理',
                 '应用经济学', '中医学/中药学', '天文学', '机械工程', '土木工程', '光学工程', '地理学', '农业资源利用', '生物学/生物科学与工程', '兵器科学与技术', '矿业工程',
                 '大气科学', '基础医学/临床医学', '电子科学与技术', '测绘科学与技术', '控制科学与工程', '军事学', '中国语言文学', '新闻传播学', '社会学', '地球物理学',
                 '植物保护'],
                ['材料科学与工程类论文', '作物学类论文', '口腔医学类论文', '药学类论文', '教育学类论文', '水利工程类论文', '理论经济学类论文', '食品科学与工程类论文',
                 '畜牧学/兽医学类论文',
                 '体育学类论文', '核科学与技术类论文', '力学类论文', '园艺学类论文', '水产类论文', '法学类论文', '地质学/地质资源与地质工程类论文', '石油与天然气工程类论文',
                 '农林经济管理类论文', '信息与通信工程类论文', '图书馆、情报与档案管理类论文', '政治学类论文', '电气工程类论文', '海洋科学类论文', '民族学类论文', '航空宇航科学与技术类论文',
                 '化学/化学工程与技术类论文', '哲学类论文', '公共卫生与预防医学类论文', '艺术学类论文', '农业工程类论文', '船舶与海洋工程类论文', '计算机科学与技术类论文', '冶金工程类论文',
                 '交通运输工程类论文', '动力工程及工程热物理类论文', '纺织科学与工程类论文', '建筑学类论文', '环境科学与工程类论文', '公共管理类论文', '数学类论文', '物理学类论文',
                 '林学/林业工程类论文', '心理学类论文', '历史学类论文', '工商管理类论文', '应用经济学类论文', '中医学/中药学类论文', '天文学类论文', '机械工程类论文', '土木工程类论文',
                 '光学工程类论文', '地理学类论文', '农业资源利用类论文', '生物学/生物科学与工程类论文', '兵器科学与技术类论文', '矿业工程类论文', '大气科学类论文', '基础医学/临床医学类论文',
                 '电子科学与技术类论文', '测绘科学与技术类论文', '控制科学与工程类论文', '军事学类论文', '中国语言文学类论文', '新闻传播学类论文', '社会学类论文', '地球物理学类论文',
                 '植物保护类论文'],
                ['这是一篇材料科学与工程类论文', '这是一篇作物学类论文', '这是一篇口腔医学类论文', '这是一篇药学类论文', '这是一篇教育学类论文', '这是一篇水利工程类论文',
                 '这是一篇理论经济学类论文',
                 '这是一篇食品科学与工程类论文', '这是一篇畜牧学/兽医学类论文', '这是一篇体育学类论文', '这是一篇核科学与技术类论文', '这是一篇力学类论文', '这是一篇园艺学类论文',
                 '这是一篇水产类论文',
                 '这是一篇法学类论文', '这是一篇地质学/地质资源与地质工程类论文', '这是一篇石油与天然气工程类论文', '这是一篇农林经济管理类论文', '这是一篇信息与通信工程类论文',
                 '这是一篇图书馆、情报与档案管理类论文', '这是一篇政治学类论文', '这是一篇电气工程类论文', '这是一篇海洋科学类论文', '这是一篇民族学类论文', '这是一篇航空宇航科学与技术类论文',
                 '这是一篇化学/化学工程与技术类论文', '这是一篇哲学类论文', '这是一篇公共卫生与预防医学类论文', '这是一篇艺术学类论文', '这是一篇农业工程类论文', '这是一篇船舶与海洋工程类论文',
                 '这是一篇计算机科学与技术类论文', '这是一篇冶金工程类论文', '这是一篇交通运输工程类论文', '这是一篇动力工程及工程热物理类论文', '这是一篇纺织科学与工程类论文', '这是一篇建筑学类论文',
                 '这是一篇环境科学与工程类论文', '这是一篇公共管理类论文', '这是一篇数学类论文', '这是一篇物理学类论文', '这是一篇林学/林业工程类论文', '这是一篇心理学类论文',
                 '这是一篇历史学类论文',
                 '这是一篇工商管理类论文', '这是一篇应用经济学类论文', '这是一篇中医学/中药学类论文', '这是一篇天文学类论文', '这是一篇机械工程类论文', '这是一篇土木工程类论文',
                 '这是一篇光学工程类论文', '这是一篇地理学类论文', '这是一篇农业资源利用类论文', '这是一篇生物学/生物科学与工程类论文', '这是一篇兵器科学与技术类论文', '这是一篇矿业工程类论文',
                 '这是一篇大气科学类论文', '这是一篇基础医学/临床医学类论文', '这是一篇电子科学与技术类论文', '这是一篇测绘科学与技术类论文', '这是一篇控制科学与工程类论文', '这是一篇军事学类论文',
                 '这是一篇中国语言文学类论文', '这是一篇新闻传播学类论文', '这是一篇社会学类论文', '这是一篇地球物理学类论文', '这是一篇植物保护类论文']]

        elif (dataset_name == 'iflytek'):
            self.train_path = r"./datasets/clue_datasets/iflytek/train_few_all.json"
            self.dev_path = r"./datasets/clue_datasets/iflytek/dev_few_all.json"
            self.test_path = r"./datasets/clue_datasets/iflytek/test_public.json"
            self.label_text2label_id = labels = {  # 中文标签对应的ID
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

            self.patterns = [
                ['打车', '地图导航', '免费WIFI', '租车', '同城服务', '快递物流', '婚庆', '家政', '公共交通', '政务', '社区服务', '薅羊毛', '魔幻', '仙侠',
                 '卡牌', '飞行空战', '射击游戏', '休闲益智', '动作类', '体育竞技', '棋牌中心', '经营养成', '策略', 'MOBA', '辅助工具', '约会社交', '即时通讯',
                 '工作社交', '论坛圈子', '婚恋社交', '情侣社交', '社交工具', '生活社交', '微博博客', '新闻', '漫画', '小说', '技术', '教辅', '问答交流', '搞笑',
                 '杂志', '百科', '影视娱乐', '求职', '兼职', '视频', '短视频', '音乐', '直播', '电台', 'K歌', '成人', '中小学', '职考', '公务员', '英语',
                 '视频教育', '高等教育', '成人教育', '艺术', '语言(非英语)', '旅游资讯', '综合预定', '民航', '铁路', '酒店', '行程管理', '民宿短租', '出国', '工具',
                 '亲子儿童', '母婴', '驾校', '违章', '汽车咨询', '汽车交易', '日常养车', '行车辅助', '租房', '买房', '装修家居', '电子产品', '问诊挂号', '养生保健',
                 '医疗服务', '减肥瘦身', '美妆美业', '菜谱', '餐饮店', '体育咨讯', '运动健身', '支付', '保险', '股票', '借贷', '理财', '彩票', '记账', '银行',
                 '美颜', '影像剪辑', '摄影修图', '相机', '绘画', '二手', '电商', '团购', '外卖', '电影票务', '社区超市', '购物咨询', '笔记', '办公', '日程管理',
                 '女性', '经营', '收款', '其他'],
                ['打车类软件', '地图导航类软件', '免费WIFI类软件', '租车类软件', '同城服务类软件', '快递物流类软件', '婚庆类软件', '家政类软件', '公共交通类软件', '政务类软件',
                 '社区服务类软件', '薅羊毛类软件', '魔幻类软件', '仙侠类软件', '卡牌类软件', '飞行空战类软件', '射击游戏类软件', '休闲益智类软件', '动作类类软件', '体育竞技类软件',
                 '棋牌中心类软件', '经营养成类软件', '策略类软件', 'MOBA类软件', '辅助工具类软件', '约会社交类软件', '即时通讯类软件', '工作社交类软件', '论坛圈子类软件',
                 '婚恋社交类软件', '情侣社交类软件', '社交工具类软件', '生活社交类软件', '微博博客类软件', '新闻类软件', '漫画类软件', '小说类软件', '技术类软件', '教辅类软件',
                 '问答交流类软件', '搞笑类软件', '杂志类软件', '百科类软件', '影视娱乐类软件', '求职类软件', '兼职类软件', '视频类软件', '短视频类软件', '音乐类软件', '直播类软件',
                 '电台类软件', 'K歌类软件', '成人类软件', '中小学类软件', '职考类软件', '公务员类软件', '英语类软件', '视频教育类软件', '高等教育类软件', '成人教育类软件',
                 '艺术类软件', '语言(非英语)类软件', '旅游资讯类软件', '综合预定类软件', '民航类软件', '铁路类软件', '酒店类软件', '行程管理类软件', '民宿短租类软件', '出国类软件',
                 '工具类软件', '亲子儿童类软件', '母婴类软件', '驾校类软件', '违章类软件', '汽车咨询类软件', '汽车交易类软件', '日常养车类软件', '行车辅助类软件', '租房类软件',
                 '买房类软件', '装修家居类软件', '电子产品类软件', '问诊挂号类软件', '养生保健类软件', '医疗服务类软件', '减肥瘦身类软件', '美妆美业类软件', '菜谱类软件',
                 '餐饮店类软件', '体育咨讯类软件', '运动健身类软件', '支付类软件', '保险类软件', '股票类软件', '借贷类软件', '理财类软件', '彩票类软件', '记账类软件', '银行类软件',
                 '美颜类软件', '影像剪辑类软件', '摄影修图类软件', '相机类软件', '绘画类软件', '二手类软件', '电商类软件', '团购类软件', '外卖类软件', '电影票务类软件',
                 '社区超市类软件', '购物咨询类软件', '笔记类软件', '办公类软件', '日程管理类软件', '女性类软件', '经营类软件', '收款类软件', '其他类软件'],
                ['这是一款打车类软件', '这是一款地图导航类软件', '这是一款免费WIFI类软件', '这是一款租车类软件', '这是一款同城服务类软件', '这是一款快递物流类软件', '这是一款婚庆类软件',
                 '这是一款家政类软件', '这是一款公共交通类软件', '这是一款政务类软件', '这是一款社区服务类软件', '这是一款薅羊毛类软件', '这是一款魔幻类软件', '这是一款仙侠类软件',
                 '这是一款卡牌类软件', '这是一款飞行空战类软件', '这是一款射击游戏类软件', '这是一款休闲益智类软件', '这是一款动作类类软件', '这是一款体育竞技类软件', '这是一款棋牌中心类软件',
                 '这是一款经营养成类软件', '这是一款策略类软件', '这是一款MOBA类软件', '这是一款辅助工具类软件', '这是一款约会社交类软件', '这是一款即时通讯类软件', '这是一款工作社交类软件',
                 '这是一款论坛圈子类软件', '这是一款婚恋社交类软件', '这是一款情侣社交类软件', '这是一款社交工具类软件', '这是一款生活社交类软件', '这是一款微博博客类软件', '这是一款新闻类软件',
                 '这是一款漫画类软件', '这是一款小说类软件', '这是一款技术类软件', '这是一款教辅类软件', '这是一款问答交流类软件', '这是一款搞笑类软件', '这是一款杂志类软件',
                 '这是一款百科类软件', '这是一款影视娱乐类软件', '这是一款求职类软件', '这是一款兼职类软件', '这是一款视频类软件', '这是一款短视频类软件', '这是一款音乐类软件',
                 '这是一款直播类软件', '这是一款电台类软件', '这是一款K歌类软件', '这是一款成人类软件', '这是一款中小学类软件', '这是一款职考类软件', '这是一款公务员类软件',
                 '这是一款英语类软件', '这是一款视频教育类软件', '这是一款高等教育类软件', '这是一款成人教育类软件', '这是一款艺术类软件', '这是一款语言(非英语)类软件', '这是一款旅游资讯类软件',
                 '这是一款综合预定类软件', '这是一款民航类软件', '这是一款铁路类软件', '这是一款酒店类软件', '这是一款行程管理类软件', '这是一款民宿短租类软件', '这是一款出国类软件',
                 '这是一款工具类软件', '这是一款亲子儿童类软件', '这是一款母婴类软件', '这是一款驾校类软件', '这是一款违章类软件', '这是一款汽车咨询类软件', '这是一款汽车交易类软件',
                 '这是一款日常养车类软件', '这是一款行车辅助类软件', '这是一款租房类软件', '这是一款买房类软件', '这是一款装修家居类软件', '这是一款电子产品类软件', '这是一款问诊挂号类软件',
                 '这是一款养生保健类软件', '这是一款医疗服务类软件', '这是一款减肥瘦身类软件', '这是一款美妆美业类软件', '这是一款菜谱类软件', '这是一款餐饮店类软件', '这是一款体育咨讯类软件',
                 '这是一款运动健身类软件', '这是一款支付类软件', '这是一款保险类软件', '这是一款股票类软件', '这是一款借贷类软件', '这是一款理财类软件', '这是一款彩票类软件',
                 '这是一款记账类软件', '这是一款银行类软件', '这是一款美颜类软件', '这是一款影像剪辑类软件', '这是一款摄影修图类软件', '这是一款相机类软件', '这是一款绘画类软件',
                 '这是一款二手类软件', '这是一款电商类软件', '这是一款团购类软件', '这是一款外卖类软件', '这是一款电影票务类软件', '这是一款社区超市类软件', '这是一款购物咨询类软件',
                 '这是一款笔记类软件', '这是一款办公类软件', '这是一款日程管理类软件', '这是一款女性类软件', '这是一款经营类软件', '这是一款收款类软件', '这是一款其他类软件']]

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
                    D.append((text , int(label)))

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

        elif (self.dataset_name == "duel2.0"):
            with open(filename, encoding='utf-8')as f:
                for l in f:
                    D.append(json.loads(l))

        # Shuffle the dataset.
        if (is_shuffle):
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
