# coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

import time

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# 查看特征列类型分布
data.dtypes.value_counts()

# 查看具体哪些特征列是object类型
data.dtypes[data.dtypes=='object']

# 删除无关特征
data=data.drop(cols,axis=1)

# 删除无关数据
data=data[(data.user_watch>=0.1)&(data.user_watch<=1)]

ids=data[0].value_counts()[data[0].value_counts()>3].index
data1=data[data[0].isin(list(ids))]
data2=data[~data[0].isin(list(ids))]

# 检查缺失值
data.isnull().sum().sort_values()

# 简单均值填充
df=df.fillna(df.mean())
	
# 字符串类型one-hot编码
data=pd.concat([data,pd.get_dummies(data['影片类型'])],axis=1)

# 标签编码
data['district']=preprocessing.LabelEncoder().fit_transform(data['地区'])

# 特征列含中文，当对中文字符串处理时，经常报错，需要做以下处理，现将特征列Unicode编码，再对单个字符串编码encode('utf-8'),然后才能转化为str类型：
data['evs_director']=data['evs_director'].astype(unicode)											#unicode
data['evs_director']=data['evs_director'].map(lambda x:' '.join(x.encode('utf-8').split('/')))		#encode('utf-8')
data['evs_director']=data['evs_director'].astype(str)

# 多值层次标签，比如中国/成都/武侯区，可以进行字段提取
data['len_location']=data['location'].map(lambda x: len(str(x).split('/')))
for i in range(3):
	data['location'+str(i)]=preprocessing.LabelEncoder().fit_transform(data['location'].map(lambda x:str(str(x).split('/')[i]) if len(str(x).split('/')) > i else ''))

# 处理时间，10位时间戳转日期、年份、月等(13位时间戳/1000)
'''
data['show_time']=data['time'].apply(timestamp_datetime)
data['show_time']=pd.to_datetime(data.show_time)
data['year']=data.show_time.dt.year
'''
def timestamp_datetime(value):
    value=time.localtime(value)
    return time.strftime('%Y-%m-%d %H:%M:%S',value)

# 划分训练集和测试集
y=data.pop('label')
train,test,train_y,test_y=train_test_split(data,y,test_size=0.3,random_state=0)


# 异常值检测
def outlier(df):
	# 基于聚类的小簇划分法及离群点划分法

	# 1.首先确定最佳簇的个数	
	
	# 1.1聚类+手肘法
	from sklearn.cluster import KMeans  
	# 存放每次结果的误差平方和  
	SSE = []  
	for k in range(1,10):  
		estimator = KMeans(n_clusters=k)  # 构造聚类器  
		estimator.fit(df)  
		SSE.append(estimator.inertia_)  
	X = range(1,10)  
	plt.xlabel('k')  
	plt.ylabel('SSE')  
	plt.plot(X,SSE,'o-')  
	plt.show()  

	# 1.2聚类+轮廓系数
	from sklearn.metrics import silhouette_score
	# 存放轮廓系数  
	Scores = []  
	for k in range(2,10):  
		estimator = KMeans(n_clusters=k)  # 构造聚类器  
		estimator.fit(df)  
		Scores.append(silhouette_score(df,estimator.labels_,metric='euclidean'))  
	X = range(2,10)  
	plt.xlabel('k')  
	plt.ylabel(u'轮廓系数')  
	plt.plot(X,Scores,'o-')  
	plt.show()  
	
	# 2.建立模型
	k=4
	iteration=500
	model=KMeans(n_clusters=k,max_iter=iteration)
	model.fit(df)
	
	# 3.判断异常点
	
	# 3.1丢离远离其他簇的小簇
	print pd.Series(model.labels_).value_counts()
	pd.DataFrame(model.cluster_centers_)
	
	# 3.2基于离群点得分检测异常点
	threshold=2
	df1=pd.concat([df,pd.Series(model.labels_,index=df.index)],axis=1)
	df1.columns=list(df.columns)+['cluster']
	norm=[]
	for i in range(k):
		norm_tmp=df1[[x for x in df1.columns if x not in 'cluster']][df1.cluster==i]-model.cluster_centers_[i]
		# 求出相对距离
		norm_tmp=norm_tmp.apply(np.linalg.norm,axis=1)
		# 求出绝对距离，相对距离/所以样本点到质心的相对距离的中位数
		norm.append(norm_tmp/norm_tmp.median())
	norm=pd.concat(norm)
	
	# 4.删除异常点
	outlier_index=norm[norm>threshold].index
	normal_index=[x for x in list(df.index) if x not in list(outlier_index)]
	df2=df.loc[normal_index,:]
	train_y1=df2.pop("label")
	
	

# 过采样和欠采样
def sampling(train,train_y):
	# 过抽样处理库SMOTE
	from imblearn.over_sampling import SMOTE 

	# 建立SMOTE模型对象
	model_smote = SMOTE() 
	# 输入数据并作过抽样处理
	x_smote_resampled, y_smote_resampled = model_smote.fit_sample(train,train_y) 
	x_smote_resampled = pd.DataFrame(x_smote_resampled, columns=train.columns)
	y_smote_resampled = pd.DataFrame(y_smote_resampled,columns=['label'])

	# 欠抽样处理库RandomUnderSampler
	from imblearn.under_sampling import RandomUnderSampler

	# 建立RandomUnderSampler模型对象
	model_RandomUnderSampler = RandomUnderSampler() 
	# 输入数据并作过抽样处理
	x_RandomUnderSampler_resampled, y_RandomUnderSampler_resampled =model_RandomUnderSampler.fit_sample(train,train_y)

	

# 单一特征离散化
def discretize(data,col,k):

	# 等值均分
	data[col]=pd.cut(data[col],k,labels=range(k))

	# 等量均分
	w=[1.0*i/k for i in range(k+1)]
	w=data[col].describe(percentiles=w)[4:4+k+1]
	data[col]=pd.cut(data[col],w,labels=range(k))
	data[col]=data[col].astype(np.float64)

	# 聚类划分
	from sklearn.cluster import KMeans
	model=KMeans(n_clusters=k)
	model.fit(data[col].reshape(-1,1))
	data[col]=pd.DataFrame(model.labels_)

