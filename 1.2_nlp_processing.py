from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# 文本处理之前的一些工作
index1=([1])
def f1(x):
    temp1=x.split(' ')
    # 求交集
    temp2=list( (set(temp1)) & (set(list(index1))) )
    if (len(temp2)==0) | (x=='nan'):
        return '0'
    else:
        return x
		
# 提取导演
data['evs_director']=data['evs_director'].astype(unicode)
data['evs_director']=data['evs_director'].map(lambda x:x.encode('utf-8').replace('/',' '))
data['evs_director']=data['evs_director'].astype(str)# 此时唯一的区别就是，以前'/'分隔符变成了' '分隔符，每一行和以前一样，含多个元素
# str1 将该列所有元素变成一个字符串，元素之间用空格隔开
str1=' '.join(data['evs_director'].tolist())
# list1将str1变成一个列表，列表每个元素代表一个词语，含重复值
list1=str1.split(' ')
# 将词频大于一定值的元素筛选出来，低于一定值的pass(在本题是6)
s1=pd.Series(list1)
index1=s1.value_counts()[s1.value_counts()>6].index
# 对该字段Series进行处理，每行词语们如果和index1有交集，则置1，否则0
data['evs_director']=data['evs_director'].map(f1)
list1=list(data['evs_director'])

# 用CountVectorizer()计算词频
cv=CountVectorizer()
# CountVectorizer默认'.'和' '都是分隔词
cv_fit=cv.fit_transform(list1)
df=pd.DataFrame(cv_fit.toarray())
data=pd.concat([data,df],axis=1)

# 用TfidfVectorizer()计算TF-IDF
cv=TfidfVectorizer()
cv_fit=cv.fit_transform(list1)
df=pd.DataFrame(cv_fit.toarray())
data=pd.concat([data,df],axis=1)
