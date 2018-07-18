from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder

# GDBT生成特征
X, y = make_classification(n_samples=10) 
train, val, train_y, val_y = train_test_split(X, y, test_size=0.3)
gbc = GradientBoostingClassifier(n_estimators=2)
one_hot = OneHotEncoder()
gbc.fit(train, train_y)
train_new = one_hot.fit_transform(gbc.apply(train)[:, :, 0])
print (train_new.todense())

# 排序特征(常作为SVM的输入)
df['category1'].rank(ascending=False, method='max')

# 缺失值统计特征
df['x_null']=
