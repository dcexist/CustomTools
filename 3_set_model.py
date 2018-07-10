from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.externals import joblib

# SVM调参
param1={'C':range(1,10,1)}
gsearch1=GridSearchCV(estimator=SVC(),param_grid=param1,scoring='accuracy',n_jobs=4,iid=False,cv=5)
gsearch1.fit(train,train_y)
gsearch1.grid_scores_,gsearch1.best_params_,gsearch1.best_score_

# Random Forest 调参

# 调节参数n_estimators
param_test1 = {'n_estimators':range(75,90,1)}
gsearch1 = GridSearchCV(estimator = RF(random_state=0), param_grid = param_test1, scoring='recall',iid=False,cv=cv)
gsearch1.fit(train,train_y)
gsearch1.grid_scores_,gsearch1.best_score_,gsearch1.best_params_

# 调节参数max_depth和min_samples_split
param_test2 = {'max_depth':range(3,9,2), 'min_samples_split':range(2,503,100)}
gsearch2 = GridSearchCV(estimator =RF(n_estimators=gsearch1.best_params_['n_estimators'],random_state=0), param_grid = param_test2, scoring='recall',iid=False, cv=cv)
gsearch2.fit(train,train_y)
gsearch2.grid_scores_,gsearch2.best_score_,gsearch2.best_params_

# 调节参数min_samples_split和min_samples_leaf
param_test3 = {'min_samples_split':range(2,200,50), 'min_samples_leaf':range(1,100,10)}
gsearch3 = GridSearchCV(estimator = RF(n_estimators=gsearch1.best_params_['n_estimators'],max_depth=gsearch2.best_params_['max_depth'], random_state=0), param_grid = param_test3, scoring='recall',iid=False, cv=cv)
gsearch3.fit(train,train_y)
gsearch3.grid_scores_,gsearch3.best_score_,gsearch3.best_params_

# 调节参数max_features
param_test4 = {'max_features':range(1,train.shape[1]+1,2)}
gsearch4 = GridSearchCV(estimator = RF( n_estimators=gsearch1.best_params_['n_estimators'],max_depth=gsearch2.best_params_['max_depth'], min_samples_leaf =gsearch3.best_params_['min_samples_leaf'], min_samples_split =gsearch3.best_params_['min_samples_split'], random_state=0), param_grid = param_test4, scoring='recall',iid=False, cv=cv)
gsearch4.fit(train,train_y)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

# 调参后的Random Forest
model=RF(n_estimators=gsearch1.best_params_['n_estimators'],max_depth=gsearch2.best_params_['max_depth'], min_samples_leaf =gsearch3.best_params_['min_samples_leaf'], min_samples_split =gsearch3.best_params_['min_samples_split'],max_features=gsearch4.best_params_['max_features'],random_state=0)
model.fit(train,train_y)
pred=model.predict(test)
metrics.recall_score(test_y,pred)




# GBDT 调参
from sklearn.model_selection import GridSearchCV

# 调节参数n_estimators
param_test1 = {'n_estimators':range(75,90,1)}
gsearch1 = GridSearchCV(estimator = GBC(learning_rate=0.1,random_state=0), param_grid = param_test1, scoring='recall',iid=False,cv=cv)
gsearch1.fit(train,train_y)
gsearch1.grid_scores_,gsearch1.best_score_,gsearch1.best_params_

# 调节参数max_depth和min_samples_split
param_test2 = {'max_depth':range(3,9,2), 'min_samples_split':range(2,503,100)}
gsearch2 = GridSearchCV(estimator =GBC(learning_rate=0.1, n_estimators=gsearch1.best_params_['n_estimators'],random_state=0), param_grid = param_test2, scoring='recall',iid=False, cv=cv)
gsearch2.fit(train,train_y)
gsearch2.grid_scores_,gsearch2.best_score_,gsearch2.best_params_

# 调节参数min_samples_split和min_samples_leaf
param_test3 = {'min_samples_split':range(2,200,50), 'min_samples_leaf':range(1,100,10)}
gsearch3 = GridSearchCV(estimator = GBC(learning_rate=0.1, n_estimators=gsearch1.best_params_['n_estimators'],max_depth=gsearch2.best_params_['max_depth'], random_state=0), param_grid = param_test3, scoring='recall',iid=False, cv=cv)
gsearch3.fit(train,train_y)
gsearch3.grid_scores_,gsearch3.best_score_,gsearch3.best_params_

# 调节参数max_features
param_test4 = {'max_features':range(7,20,2)}
gsearch4 = GridSearchCV(estimator = GBC(learning_rate=0.1, n_estimators=gsearch1.best_params_['n_estimators'],max_depth=gsearch2.best_params_['max_depth'], min_samples_leaf =gsearch3.best_params_['min_samples_leaf'], min_samples_split =gsearch3.best_params_['min_samples_split'], random_state=0), param_grid = param_test4, scoring='recall',iid=False, cv=cv)
gsearch4.fit(train,train_y)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

# 调节参数subsample
param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch5 = GridSearchCV(estimator = GBC(learning_rate=0.1, n_estimators=gsearch1.best_params_['n_estimators'],max_depth=gsearch2.best_params_['max_depth'], min_samples_leaf =gsearch3.best_params_['min_samples_leaf'], min_samples_split =gsearch3.best_params_['min_samples_split'], max_features=gsearch4.best_params_['max_features'],random_state=0), param_grid = param_test5, scoring='recall',iid=False, cv=cv)
gsearch4.fit(train,train_y)
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_

# 调参后的GDBT,这时可以通过减半步长，n_estimators加倍等调整来增加模型的泛化能力
model=GBC(learning_rate=0.1/2, n_estimators=gsearch1.best_params_['n_estimators']*2,max_depth=gsearch2.best_params_['max_depth'], min_samples_leaf =gsearch3.best_params_['min_samples_leaf'],min_samples_split =gsearch3.best_params_['min_samples_split'], max_features=gsearch4.best_params_['max_features'],subsample=gsearch5.best_params_['subsample'],random_state=0)
model.fit(train,train_y)
pred=model.predict(test)
metrics.recall_score(test_y,pred)




# stacking 模型融合
def ensemble_model(model,train,train_y,test,n_folds=5,random_state=0):
    
    num_train, num_test = train.shape[0], test.shape[0]
    L1_train = np.zeros((num_train,)) 
    L1_test = np.zeros((num_test,))
    L1_test_all = np.zeros((num_test, n_folds))
    KF = KFold(n_splits = n_folds, random_state=random_state)
    
    for i, (train_index, val_index) in enumerate(KF.split(train)):
        x_train, y_train = train[train_index], train_y[train_index]
        x_val, y_val = train[val_index], train_y[val_index]
        model.fit(x_train,y_train)
        L1_train[val_index] = model.predict(x_val)
        L1_test_all[:, i] = model.predict(test)
    L1_test = np.mean(L1_test_all, axis=1)
    
    return L1_train,L1_test
	
model=GBC(learning_rate=0.1, n_estimators=78,max_depth=5, min_samples_leaf =1,min_samples_split =52,random_state=0)
gbc_train,gbc_test=ensemble_model(model,train.values,train_y.values,test.values)

model=RF(n_estimators=75,max_depth=7,min_samples_leaf =1, min_samples_split =2, max_features=5,random_state=0)
rf_train,rf_test=ensemble_model(model,train.values,train_y.values,test.values)

input_train=[gbc_train,rf_train] 
input_test=[gbc_test,rf_test]

stacked_train = np.concatenate([f.reshape(-1, 1) for f in input_train], axis=1)
stacked_test = np.concatenate([f.reshape(-1, 1) for f in input_test], axis=1)

model=GBC()
model.fit(stacked_train, y_train)
pred=model.predict(stacked_test)
print metrics.recall_score(test_y,pred),metrics.precision_score(test_y,pred)
joblib.dump(model,'../model/stacking_model.pkl')