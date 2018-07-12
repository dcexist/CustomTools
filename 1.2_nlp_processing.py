from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora,similarities,models  
from gensim.models import word2vec
import jieba
import lda

# 基于sklearn的TF,TF-IDF计算
index1=([1])
col='category'

def f1(x):
    temp1=x.split(' ')
    # 求交集
    temp2=list( (set(temp1)) & (set(list(index1))) )
    if (len(temp2)==0) | (x=='nan'):
        return '0'
    else:
        return x

def sk_process(data,col):		
	# 提取字段
	data[col]=data[col].astype(unicode)
	data[col]=data[col].map(lambda x:x.encode('utf-8').replace('/',' '))
	data[col]=data[col].astype(str)# 此时唯一的区别就是，以前'/'分隔符变成了' '分隔符，每一行和以前一样，含多个元素
	# str1 将该列所有元素变成一个字符串，元素之间用空格隔开
	str1=' '.join(data[col].tolist())
	# list1将str1变成一个列表，列表每个元素代表一个词语，含重复值
	list1=str1.split(' ')
	# 将词频大于一定值的元素筛选出来，低于一定值的pass(在本题是6)
	s1=pd.Series(list1)
	index1=s1.value_counts()[s1.value_counts()>6].index
	# 对该字段Series进行处理，每行词语们如果和index1有交集，则置1，否则0
	data[col]=data[col].map(f1)
	list1=list(data[col])

	# 用CountVectorizer()计算词频
	cv=CountVectorizer()
	# CountVectorizer默认'.'和' '都是分隔词
	cv_fit=cv.fit_transform(list1)
	df=pd.DataFrame(cv_fit.toarray())
	data1=pd.concat([data,df],axis=1)

	# 用TfidfVectorizer()计算TF-IDF
	cv=TfidfVectorizer()
	cv_fit=cv.fit_transform(list1)
	df=pd.DataFrame(cv_fit.toarray())
	data2=pd.concat([data,df],axis=1)
	
	return data1,data2

# 基于gensim的TF、TF-IDF和相似度计算
raw_documents = [  
'0南京江心洲污泥偷排”等污泥偷排或处置不当而造成的污染问题，不断被媒体曝光',  
'1面对美国金融危机冲击与国内经济增速下滑形势，中国政府在2008年11月初快速推出“4万亿”投资十项措施',  
'2全国大面积出现的雾霾，使解决我国环境质量恶化问题的紧迫性得到全社会的广泛关注',  
'3大约是1962年的夏天吧，潘文突然出现在我们居住的安宁巷中，她旁边走着40号王孃孃家的大儿子，一看就知道，他们是一对恋人。那时候，潘文梳着一条长长的独辫',  
'4坐落在美国科罗拉多州的小镇蒙特苏马有一座4200平方英尺(约合390平方米)的房子，该建筑外表上与普通民居毫无区别，但其内在构造却别有洞天',  
'5据英国《每日邮报》报道，美国威斯康辛州的非营利组织“占领麦迪逊建筑公司”(OMBuild)在华盛顿和俄勒冈州打造了99平方英尺(约9平方米)的迷你房屋',  
'6长沙市公安局官方微博@长沙警事发布消息称，3月14日上午10时15分许，长沙市开福区伍家岭沙湖桥菜市场内，两名摊贩因纠纷引发互殴，其中一人被对方砍死',  
'7乌克兰克里米亚就留在乌克兰还是加入俄罗斯举行全民公投，全部选票的统计结果表明，96.6%的选民赞成克里米亚加入俄罗斯，但未获得乌克兰和国际社会的普遍承认',  
'8京津冀的大气污染，造成了巨大的综合负面效应，显性的是空气污染、水质变差、交通拥堵、食品不安全等，隐性的是各种恶性疾病的患者增加，生存环境越来越差',  
'9 1954年2月19日，苏联最高苏维埃主席团，在“兄弟的乌克兰与俄罗斯结盟300周年之际”通过决议，将俄罗斯联邦的克里米亚州，划归乌克兰加盟共和国',  
'10北京市昌平区一航空训练基地，演练人员身穿训练服，从机舱逃生门滑降到地面',  
'11腾讯入股京东的公告如期而至，与三周前的传闻吻合。毫无疑问，仅仅是传闻阶段的“联姻”，已经改变了京东赴美上市的舆论氛围',  
'12国防部网站消息，3月8日凌晨，马来西亚航空公司MH370航班起飞后与地面失去联系，西安卫星测控中心在第一时间启动应急机制，配合地面搜救人员开展对失联航班的搜索救援行动',  
'13新华社昆明3月2日电，记者从昆明市政府新闻办获悉，昆明“3·01”事件事发现场证据表明，这是一起由新疆分裂势力一手策划组织的严重暴力恐怖事件',  
'14在即将召开的全国“两会”上，中国政府将提出2014年GDP增长7.5%左右、CPI通胀率控制在3.5%的目标',  
'15中共中央总书记、国家主席、中央军委主席习近平看望出席全国政协十二届二次会议的委员并参加分组讨论时强调，团结稳定是福，分裂动乱是祸。全国各族人民都要珍惜民族大团结的政治局面，都要坚决反对一切危害各民族大团结的言行'  
]

def ge_process(raw_documents):

	corpora_documents = []  
	#分词处理  
	for item_text in raw_documents:  
	item_seg = list(jieba.cut(item_text))  
	corpora_documents.append(item_seg)  
	# 生成字典语料  
	dictionary = corpora.Dictionary(corpora_documents)

	# 词频统计,稀疏表达方式，实际上产生的是16*384的词频矩阵，16是文档数目，384是词语数目
		#dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=100000) 
			# 1.去掉出现次数低于no_below的 
			# 2.去掉出现次数高于no_above的。注意这个小数指的是百分数 
			# 3.在1和2的基础上，保留出现频率前keep_n的单词
	corpus = [dictionary.doc2bow(text) for text in corpora_documents]  
	# 计算TF-IDF
	tfidf_model = models.TfidfModel(corpus)  
	corpus_tfidf = tfidf_model[corpus]  
	# 计算相似度
	similarity = similarities.Similarity('Similarity-tfidf-index', corpus_tfidf, num_features=600)  
	test_data_1 = '北京雾霾红色预警'  
	test_cut_raw_1 = list(jieba.cut(test_data_1))  # ['北京', '雾', '霾', '红色', '预警']  
	test_corpus_1 = dictionary.doc2bow(test_cut_raw_1)  # [(51, 1), (59, 1)]，即在字典的56和60的地方出现重复的字段，这个值可能会变化  
	# 定义相似样本数量N
	similarity.num_best = 5  
	test_corpus_tfidf_1=tfidf_model[test_corpus_1]  # 根据之前训练生成的model，生成query的IFIDF值，然后进行相似度计算  
													# [(51, 0.7071067811865475), (59, 0.7071067811865475)]  
	print(similarity[test_corpus_tfidf_1])  # [(2, 0.3595932722091675)]

	# 利用潜在语义序列计算相似度，先获得tf-idf(也可以直接使用bow向量)
	lsi = models.LsiModel(corpus_tfidf)  
	corpus_lsi = lsi[corpus_tfidf]  
	similarity_lsi=similarities.Similarity('Similarity-LSI-index', corpus_lsi, num_features=400,num_best=2)  
	test_data_3 = '长沙街头发生砍人事件致6人死亡'  
	test_cut_raw_3 = list(jieba.cut(test_data_3))         
	test_corpus_3 = dictionary.doc2bow(test_cut_raw_3)  
	test_corpus_tfidf_3 = tfidf_model[test_corpus_3]   
	test_corpus_lsi_3 = lsi[test_corpus_tfidf_3]  
	print(similarity_lsi[test_corpus_lsi_3])    

# 基于lda的主题模型
# 词频矩阵，(395L, 4258L)，表示395个文档，4258个单词 
X = lda.datasets.load_reuters()

def lda_process(X)
	
	# 建立LDA模型
	model = lda.LDA(n_topics=20, n_iter=500, random_state=1)  
	model.fit(X)
	# 主题-单词分布,shape: (20L, 4258L)
	topic_word = model.topic_word_
	# 计算各主题Top-N个单词
	n = 5
	for i, topic_dist in enumerate(topic_word):
		topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
		print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))
	# 文档-主题分布，shape: (395L, 20L)
	doc_topic = model.doc_topic_
	return doc_topic
	
# 基于gensim的相似度计算
# 便于更好的将名字区分
jieba.suggest_freq('沙瑞金', True)
jieba.suggest_freq('田国富', True)
jieba.suggest_freq('高育良', True)
jieba.suggest_freq('侯亮平', True)
jieba.suggest_freq('钟小艾', True)
jieba.suggest_freq('陈岩石', True)
jieba.suggest_freq('欧阳菁', True)
jieba.suggest_freq('易学习', True)
jieba.suggest_freq('王大路', True)
jieba.suggest_freq('蔡成功', True)
jieba.suggest_freq('孙连城', True)
jieba.suggest_freq('季昌明', True)
jieba.suggest_freq('丁义珍', True)
jieba.suggest_freq('郑西坡', True)
jieba.suggest_freq('赵东来', True)
jieba.suggest_freq('高小琴', True)
jieba.suggest_freq('赵瑞龙', True)
jieba.suggest_freq('林华华', True)
jieba.suggest_freq('陆亦可', True)
jieba.suggest_freq('刘新建', True)
jieba.suggest_freq('刘庆祝', True)

with open('./data/in_the_name_of_people.txt') as f:
    
    document = f.read()
    document_cut = jieba.cut(document)
    result = ' '.join(document_cut)
    result = result.encode('utf-8')
    with open('./data/in_the_name_of_people_segment.txt', 'w') as f2:
        f2.write(result)
f.close()
f2.close()

def ge_word2vec():
	# LineSentence类来读文件
	sentences = word2vec.LineSentence('./data/in_the_name_of_people_segment.txt') 
	# size: 词向量的维度
	# window：即词向量上下文最大距离
	# min_count:需要计算词向量的最小词频
	model = word2vec.Word2Vec(sentences, hs=1,min_count=1,window=3,size=100)  
	
	# 1.找出某一个词向量最相近的词集合
	req_count = 5
	for key in model.wv.similar_by_word('沙瑞金'.decode('utf-8'), topn =100):
		if len(key[0])==3:
			req_count -= 1
			print key[0], key[1]
			if req_count == 0:
				break;
	# 高育良 0.957465648651
	# 李达康 0.956072568893
	# 田国富 0.949632883072
	# 易学习 0.941760063171
	# 侯亮平 0.938041090965
	
	# 2.两个词向量的相近程度
	print model.wv.similarity('沙瑞金'.decode('utf-8'), '高育良'.decode('utf-8'))# 0.961137455325
	# 3.找出不同类的词
	print model.wv.doesnt_match(u"沙瑞金 高育良 李达康 刘庆祝".split())# 刘庆祝