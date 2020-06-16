import zhconv
import jieba
import numpy as np
import requests
import json
import tensorflow as tf


class AspectCNN(tf.keras.Model):
    def __init__(self,
                 word_embed,
                 aspect_input,
                 kernel_num=200,
                 kernel_sizes=[3, 4],
                 n_class=4):
        super(AspectCNN, self).__init__()
        
        ############## 相关参数 #############
        # 词嵌入
        self.word_embed = word_embed
        self.vocab_size, self.embed_dim = word_embed.shape
        
        # 角度向量，每个角度都由一组单词组成
        # 角度嵌入，20个角度，每个角度表示成一个向量，然后可以分成四类
        # 如角度：位置，分成四类：方便、普通、不方便、未提及
        # 如角度：口味，分成四类：好吃、普通、难吃、未提及
        self.aspect_input = aspect_input
        self.num_aspects, _ = aspect_input.shape
        
        # CNN处理参数
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes
        
        # 四分类
        self.n_class = n_class
        
        ############### 相关层 #########################
        self.word_embedings = tf.keras.layers.Embedding(
            self.vocab_size, self.embed_dim, weights=[self.word_embed])
        
        self.aspect_fc = tf.keras.layers.Dense(self.kernel_num)
        
        self.convs4word = [
            tf.keras.layers.Conv1D(filters=self.kernel_num,
                                   kernel_size=k,
                                   activation='tanh')
            for k in self.kernel_sizes
        ]
        
        self.convs4aspect = [
            tf.keras.layers.Conv1D(filters=self.kernel_num,
                                   kernel_size=k,
                                   activation=None) for k in self.kernel_sizes
        ]
        
        self.fc = tf.keras.layers.Dense(self.n_class, activation='softmax')
    
    def get_aspect_embed(self):
        # num_aspects,aspect_words,embed_size
        aspects_embed = self.word_embedings(self.aspect_input)
        
        # num_aspects,embed_size
        aspects_embed = tf.reduce_mean(aspects_embed, axis=1)
        return aspects_embed
    
    def call(self, input_text):
        batch_size = input_text.shape[0]
        feature = self.word_embedings(input_text)  # batch,seq_len,embed_size
        
        aspects_embed = self.get_aspect_embed()  # num_aspects,embed_size
        aspects = self.aspect_fc(aspects_embed)  # num_aspects,kernel_num
        aspects = tf.expand_dims(aspects, 1)  # num_aspects,1,kernel_num
        
        x = [tf.expand_dims(conv(feature), 1) for conv in self.convs4word]
        # batch, 1, seq_len-k+1, kernel_num
        
        y = [
            tf.nn.relu(tf.expand_dims(conv(feature), 1) + aspects)
            for conv in self.convs4aspect
        ]
        # batch, num_aspects, seq_len-k+1, kernel_num
        
        x0 = [i * j for i, j in zip(x, y)]
        x0 = [tf.reshape(t, tf.constant((-1, *t.shape[2:]))) for t in x0]
        x0 = [
            tf.squeeze(tf.nn.max_pool1d(t, t.shape[1], 1, padding='VALID'), 1)
            for t in x0
        ]

        shape = np.array([batch_size, self.num_aspects,
                          self.kernel_num]).astype(np.int32)
        x0 = [tf.reshape(t, shape) for t in x0]
        x0 = tf.concat(x0, 2)
        return self.fc(x0)  # batch,num_aspects,n_class


def get_stop_word_set(fname, only_punctuation=True):
    words_set = set()
    with open(fname, encoding='utf-8') as f_r:
        for line in f_r:
            words_set |= set(line.strip())
    if only_punctuation:
        words_set |= set([' '])
    return words_set


def get_vocab(vocab_path):
    word2id = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            word, idx = line.split()
            idx = int(idx)
            word2id[word] = idx
    return word2id


def process(input_text, word2id, stop_words, max_len):
    content = zhconv.convert(input_text.strip(), 'zh-cn')
    content = list(
        filter(lambda x: len(x.strip()) > 0, list(jieba.cut(content))))
    ids = [word2id[word] for word in content if word not in stop_words and word in word2id]
    if max_len > len(ids):
        ids = ids + [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return np.array([ids])


class ResultRender:
    label2zh = {'location_traffic_convenience': '交通是否便利',
                'location_distance_from_business_district': '距离商圈远近',
                'location_easy_to_find': '是否容易寻找',
                'service_wait_time': '排队等候时间',
                'service_waiters_attitude': '服务人员态度',
                'service_parking_convenience': '是否容易停车',
                'service_serving_speed': '点菜/上菜速度', 'price_level': '价格水平',
                'price_cost_effective': '性价比', 'price_discount': '折扣力度',
                'environment_decoration': '装修情况', 'environment_noise': '嘈杂情况',
                'environment_space': '就餐空间', 'environment_cleaness': '卫生情况',
                'dish_portion': '分量', 'dish_taste': '口感', 'dish_look': '外观',
                'dish_recommendation': '推荐程度',
                'others_overall_experience': '本次消费感受',
                'others_willing_to_consume_again': '再次消费的意愿', 'location': '位置',
                'service': '服务', 'price': '价格', 'environment': '环境',
                'dish': '菜品', 'others': '其它'}
    
    label_layers = {'location': ['location_traffic_convenience',
                                 'location_distance_from_business_district',
                                 'location_easy_to_find'],
                    'service': ['service_wait_time', 'service_waiters_attitude',
                                'service_parking_convenience',
                                'service_serving_speed'],
                    'price': ['price_level', 'price_cost_effective',
                              'price_discount'],
                    'environment': ['environment_decoration',
                                    'environment_noise', 'environment_space',
                                    'environment_cleaness'],
                    'dish': ['dish_portion', 'dish_taste', 'dish_look',
                             'dish_recommendation'],
                    'others': ['others_overall_experience',
                               'others_willing_to_consume_again']}

    subjects_eng = [
        'location_traffic_convenience',
        'location_distance_from_business_district',
        'location_easy_to_find',
        'service_wait_time',
        'service_waiters_attitude',
        'service_parking_convenience',
        'service_serving_speed',
        'price_level',
        'price_cost_effective',
        'price_discount',
        'environment_decoration',
        'environment_noise',
        'environment_space',
        'environment_cleaness',
        'dish_portion',
        'dish_taste',
        'dish_look',
        'dish_recommendation',
        'others_overall_experience',
        'others_willing_to_consume_again',
    ]
    
    sentiment = ['未提及', '负向', '中性', '正向']
    
    def render(self, prob_dict, mode='tree'):
        if mode == 'tree':
            results = {'name': '点评', 'children': []}
            for layer1 in self.label_layers.keys():
                layer1_zh = self.label2zh[layer1]
                layer1_dict = {'name': layer1_zh, 'children': []}
                for layer2 in self.label_layers[layer1]:
                    probs = prob_dict.get(layer2, [0, 0, 0, 0])
                    label = ['未提及', '负向', '中性', '正向']
                    layer2_zh = self.label2zh[layer2]
                    prob_with_label = [{'name': l, 'value': str(round(v, 2))}
                                       for l, v in zip(label, probs)]
                    # 如何可视化这结果？？？
                    layer2_dict = {'name': layer2_zh,
                                   'children': prob_with_label}
                    layer1_dict['children'].append(layer2_dict)
                results['children'].append(layer1_dict)
            return results



    
    
    
    
    
    
    
    
    
    