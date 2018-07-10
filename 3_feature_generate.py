from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder

from gensim import corpora,similarities,models  
import jieba 

# GDBT生成特征
X, y = make_classification(n_samples=10) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
gbc = GradientBoostingClassifier(n_estimators=2)
one_hot = OneHotEncoder()
gbc.fit(X_train, y_train)
X_train_new = one_hot.fit_transform(gbc.apply(X_train)[:, :, 0])
print (X_train_new.todense())

# 排序特征(常作为SVM的输入)
df['category1'].rank(ascending=False, method='max')

# 缺失值统计特征
df['x_null']=