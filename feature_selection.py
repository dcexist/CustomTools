# coding=utf-8

from sklearn.ensemble import GradientBoostingClassifier as GBC

# 交叉验证
from sklearn.model_selection import StratifiedKFold
cv=StratifiedKFold(n_splits=5, random_state=0, shuffle=False)
cross_val_score(clf, train, train_y, cv=cv, scoring='precision').mean())


# 方差筛选
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold()
selector.fit(train)
print np.sort(selector.variances_)

def variance_select(train,train_y,a,b,step,c)：
	for i in range(a,b,step):
		selector = VarianceThreshold(threshold=i/c)
		selector.fit(train)
		train1=selector.transform(train)
		
		clf=GBC(random_state=0)
		print round(cross_val_score(clf, train1, train_y, cv=cv, scoring='recall').mean(),4)
		print "_____________________________________"

		
# 卡方检验
from sklearn.feature_selection import chi2
corrlation={}
for i in range(train.shape[1]):
    corrlation[train.columns[i]]=chi2(train,train_y)[0][i]
pd.DataFrame.from_dict(corrlation,orient='index').sort_values(by=[0],ascending=False)

def chi2_select(train,train_y):
	from sklearn.feature_selection import SelectKBest
	from sklearn.feature_selection import chi2
	score=0
	index=1
	for i in range(1,train.shape[1]+1):
		model=SelectKBest(chi2,k=i)
		train1=model.fit_transform(train,train_y)
		
		clf=GBC(random_state=0)
		cv_score=cross_val_score(clf, train1, train_y, cv=cv, scoring='recall').mean()
		if score<cv_score:
			score=cv_score
			index=i
		print i,round(cv_score,4)
	print "______________________"
	print index,score
	# 被删除的特征
	model=SelectKBest(chi2,k=index).fit(train,train_y)
	train.columns[~model.get_support()]
	
	
# 最大信息系数
from minepy import MINE
m=MINE()
cols=train.columns
corrlation={}
for col in cols:
    m.compute_score(train[col],train_y)
    corrlation[col]=m.mic()
pd.DataFrame.from_dict(corrlation,orient='index').sort_values(by=[0],ascending=False)

# 互信息法筛选
def mutual_info_select(train,train_y):
	from sklearn.feature_selection import SelectKBest
	from sklearn.feature_selection import mutual_info_classif
	score=0
	index=1
	for i in range(1,train.shape[1]+1):
		model=SelectKBest(mutual_info_classif,k=i)
		a_train=model.fit_transform(train,train_y)
		
		clf = GBC(random_state=0)
		cv_score=cross_val_score(clf, a_train, train_y, cv=cv, scoring='recall').mean()
		if score<cv_score:
			score=cv_score
			index=i
		print i,round(cv_score,4)
	print "______________________"
	print index,score
	# 被删除的特征
	model=SelectKBest(mutual_info_classif,k=index).fit(train,train_y)
	train.columns[~model.get_support()]

# 基于相关系数的假设检验
def f_classif_select(train,train_y):
	from sklearn.feature_selection import SelectKBest
	from sklearn.feature_selection import f_classif
	score=0
	index=1
	for i in range(1,train.shape[1]+1):
		model=SelectKBest(f_classif,k=i)
		train1=model.fit_transform(train,train_y)
		
		clf = GBC(random_state=0)
		cv_score=cross_val_score(clf, train1, train_y, cv=cv, scoring='recall').mean()
		if score<cv_score:
			score=cv_score
			index=i
		print i,round(cv_score,4)
	print "______________________"
	print index,score
	model=SelectKBest(f_classif,k=index).fit(train,train_y)
	train.columns[~model.get_support()]
	
# 基于GDBT的单变量特征选择
clf =GBC(random_state=0)
scores=[]
columns=train.columns
corrlation={}
for i in range(train.shape[1]):
    score=cross_val_score(clf,train.values[:,i:i+1],train_y.reshape(-1,1),scoring='recall',
                          cv=cv)
    corrlation[columns[i]]=format(np.mean(score),'.4f')
pd.DataFrame.from_dict(corrlation,orient='index').sort_values(by=[0],ascending=False)

# 递归特征消除
# 通过交叉验证自动确定消除特征数目
from sklearn.feature_selection import RFECV
clf=RFECV(estimator=GBC(random_state=0),step=1,cv=cv,scoring='recall')
clf.fit(train,train_y)
# 被消除的特征
print train.columns[~clf.support_],np.max(clf.grid_scores_)

# 基于L1的LR特征选择
def L1_select(train,train_y,a,b,step,c):
	from sklearn.feature_selection import SelectFromModel
	score=0
	index=0
	clf1=LogsiticRegreesion(penalty="l1").fit(train.values, train_y.values.reshape(-1,1))
	for i in range(a,b,step):
		model = SelectFromModel(clf1,threshold=i/c)
		model.fit(train,train_y)
		train1=model.transform(train)
		clf =GBC(random_state=0)
		cv_score=cross_val_score(clf, train1, train_y, cv=cv, scoring='recall').mean()
		if score<cv_score:
			score=cv_score
			index=i/c
		print i/c,cv_score
	print
	print index,score

clf1=LR(penalty="l1").fit(train.values, train_y.values.reshape(-1,1))
model = SelectFromModel(clf1,threshold=index)
model.fit(train,train_y)
train1=model.transform(train)

clf =GBC(random_state=0)
print train.columns[~model.get_support()]
print round(cross_val_score(clf, train1, train_y, cv=cv, scoring='recall').mean(),4)


# 基于GDBT的特征选择
def gdbt_select(train,train_y,a,b,step,c):
	from sklearn.feature_selection import SelectFromModel

	score=0
	index=0
	clf1=GBC(random_state=0).fit(train.values, train_y.values.reshape(-1,1))
	for i in range(a,b,step):
		model = SelectFromModel(clf1,threshold=i/c)
		model.fit(train,train_y)
		train1=model.transform(train)
		clf =GBC(random_state=0)
		cv_score=cross_val_score(clf, train1, train_y, cv=cv, scoring='recall').mean()
		if score<cv_score:
			score=cv_score
			index=i/c
		print i/c,cv_score
	print
	print index,score

model = SelectFromModel(clf1,threshold=index)
model.fit(train,train_y)
train1=model.transform(train)

clf =GBC(random_state=0)
print train.columns[~model.get_support()]
print round(cross_val_score(clf, train1, train_y, cv=cv, scoring='recall').mean(),4)

'''
makedown 语法：
#### 小结
- 方差筛选
    - discarded feature: None
    - recall score: 
- 卡方检验
    - discarded feature: 
    - recall score:
- 互信息法
    - discarded feature: 
    - recall score:
- 基于相关系数的假设检验
    - discarded feature: 
    - recall score:
- 基于GDBT的单变量特征选择
    - discarded feature: 
    - recall score:
- 递归特征消除
    - discarded feature:
    - recall score:
- 基于L1的LR特征选择
    - discarded feature: 
    - recall score:
- 基于GDBT的特征选择
    - discarded feature:
    - recall score:
'''