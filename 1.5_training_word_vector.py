from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 对中文词语训练词向量，保存模型
def my_function():
	# 读取语料库
    wiki_news = open('./data/reduce_zhiwiki.txt', 'r')
	# 第一个参数是预处理后的训练语料库。
	# sg=0表示使用CBOW模型训练词向量；sg=1表示利用Skip-gram训练词向量。
	# 参数size表示词向量的维度。
	# windows表示当前词和预测词可能的最大距离，windows越大所需要枚举的预测词越多，计算时间越长。
	# min_count表示最小出现的次数，如果一个词语出现的次数小于min_count，那么直接忽略该词语。
	# workers表示训练词向量时使用的线程数。
    model = Word2Vec(LineSentence(wiki_news), sg=0,size=192, window=5, min_count=5, workers=9)
    model.save('zhiwiki_news.word2vec')

if __name__ == '__main__':
    my_function()