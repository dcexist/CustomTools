#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gensim.models as g
from gensim.corpora import WikiCorpus
import logging
from langconv import *

#enable logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#读取维基百科中文预料，并训练段落向量，保存模型
docvec_size=192
class TaggedWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True
    def __iter__(self):
        import jieba
        for content, (page_id, title) in self.wiki.get_texts():
			# yield是一个类似return的关键字，迭代一次遇到yield时就返回yield后面(右边)的值。重点是：下一次迭代时，从上一次迭代遇到的yield后面的代码(下一行)开始执行。
			# 嵌套列表
            yield g.doc2vec.LabeledSentence(words=[w for c in content for w in jieba.cut(Converter('zh-hans').convert(c))], tags=[title])

def my_function():
    zhwiki_name = './data/zhwiki-latest-pages-articles.xml.bz2'
    wiki = WikiCorpus(zhwiki_name, lemmatize=False, dictionary={})
    documents = TaggedWikiDocument(wiki)

	# documents表示用于训练的语料文章。
	# dm表示训练时使用的模型种类，一般dm默认等于1，这时默认使用DM模型；当dm等于其他值时，使用DBOW模型训练词向量。
	# size代表段落向量的维度。
	# window表示当前词和预测词可能的最大距离。
	# min_count表示最小出现的次数。
	# workers表示训练词向量时使用的线程数。
    model = g.Doc2Vec(documents, dm=0, dbow_words=1, size=docvec_size, window=8, min_count=19, iter=5, workers=8)
    model.save('data/zhiwiki_news.doc2vec')

if __name__ == '__main__':
    my_function()

