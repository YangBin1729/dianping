__author__ = 'yangbin1729'


class Config:
    SECRET_KEY = '123@456#789$012%'
    
    stopwords_path = "saved/哈工大停用标点表.txt"
    vocab_path = "saved/vocab.txt"
    model_path = "saved/models"
    max_len = 1113
    
    word_embed_path = "saved/word_embed.npy"
    aspect_path = "saved/aspects.npy"
