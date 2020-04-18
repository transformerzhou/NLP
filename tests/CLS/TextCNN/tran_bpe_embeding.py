#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""
from gensim.models import Word2Vec

from fennlp.bpemd import bpe

# you can also use the corpus you have processed and set corpus ="yourfile"
# if don't use yourself corpus, you can download wiki data from
# "https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2
# pay attention: if you want to train chinese embedding, you need use chinese zh_vocab.txt,
# otherwise, use english zh_vocab.txt
#  https://dumps.wikimedia.org/zhwiki/latest/
corpus = "./corpus/zhwiki-latest-pages-articles.xml.bz2"  # could download from wiki
#  https://dumps.wikimedia.org/enwiki/latest/
# corpus = "enwiki-latest-pages-articles.xml.bz2 " # could download from wiki
vocab = "./corpus/zh_vocab.txt"  # here vocab file could download from bert

traniner = bpe.BPE(corpus=corpus,
                   vocab_files=vocab,
                   language='zh')

traniner.train_word2vec(embed_size=100,
                        window_size=5,
                        min_count=0)

# test your embedding
wiki_word2vec_model = Word2Vec.load('./corpus/word2vec.model')
testwords = ['猪', '上', '树', '猪', '在', '飞']
for i in range(5):
    res = wiki_word2vec_model.most_similar(testwords[i])
    print(testwords[i])
    print(res)
