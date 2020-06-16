__author__ = 'yangbin1729'

import numpy as np
import tensorflow as tf
from .aspect import AspectCNN, get_vocab, get_stop_word_set, ResultRender, process


def predict(config, input_text):
    stop_words = get_stop_word_set(config.stopwords_path)
    word2id = get_vocab(config.vocab_path)
    word_embed = np.load(config.word_embed_path)
    aspet_input = np.load(config.aspect_path)
    
    model = AspectCNN(word_embed, aspet_input)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(config.model_path))
    
    inputs = process(input_text, word2id, stop_words, config.max_len)
    
    proba = model(inputs).numpy()[0].tolist()  # 1,22,4
    
    result_render = ResultRender()
    
    prob_dict = {}
    for key, val in zip(result_render.subjects_eng, proba):
        prob_dict[key] = val
    
    return result_render.render(prob_dict)