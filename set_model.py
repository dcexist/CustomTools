from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RF


# GBDT调参
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
clf=GBC(learning_rate=0.1/2, n_estimators=gsearch1.best_params_['n_estimators']*2,max_depth=gsearch2.best_params_['max_depth'], min_samples_leaf =gsearch3.best_params_['min_samples_leaf'],min_samples_split =gsearch3.best_params_['min_samples_split'], max_features=gsearch4.best_params_['max_features'],subsample=gsearch5.best_params_['subsample'],random_state=0)
clf.fit(train,train_y)
pred=clf.predict(test)
metrics.recall_score(test_y,pred)


# Random Forest调参

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
param_test4 = {'max_features':range(1,20,2)}
gsearch4 = GridSearchCV(estimator = RF( n_estimators=gsearch1.best_params_['n_estimators'],max_depth=gsearch2.best_params_['max_depth'], min_samples_leaf =gsearch3.best_params_['min_samples_leaf'], min_samples_split =gsearch3.best_params_['min_samples_split'], random_state=0), param_grid = param_test4, scoring='recall',iid=False, cv=cv)
gsearch4.fit(train,train_y)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

# 调参后的Random Forest
clf=RF(n_estimators=gsearch1.best_params_['n_estimators'],max_depth=gsearch2.best_params_['max_depth'], min_samples_leaf =gsearch3.best_params_['min_samples_leaf'], min_samples_split =gsearch3.best_params_['min_samples_split'],max_features=gsearch4['max_features'],random_state=0)
clf.fit(train,train_y)
pred=clf.predict(test)
metrics.recall_score(test_y,pred)