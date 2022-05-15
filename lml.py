#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 18:26:35 2021

@author: ramil_saidashev
"""
import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from itertools import groupby
data, test_data = pd.read_csv('./train.csv'), pd.read_csv('./test.csv')

num_cols = [
    'ClientPeriod',
    'MonthlySpending',
    'TotalSpent'
]

cat_cols = [
    'Sex',
    'IsSeniorCitizen',
    'HasPartner',
    'HasChild',
    'HasPhoneService',
    'HasMultiplePhoneNumbers',
    'HasInternetService',
    'HasOnlineSecurityService',
    'HasOnlineBackup',
    'HasDeviceProtection',
    'HasTechSupportAccess',
    'HasOnlineTV',
    'HasMovieSubscription',
    'HasContractPhone',
    'IsBillingPaperless',
    'PaymentMethod'
]

feature_cols = num_cols + cat_cols
target_col = 'Churn'
data_cols = feature_cols + [target_col]

#1
data = data.replace(r'^\s*$', np.nan, regex=True)
data.replace([np.inf, -np.inf], np.nan, inplace=True)

for i in range(5):
    print(data.loc[random.choice([i for i in range(len(data))])], end = '\n*******\n\n')

s = 0   
table = data.isnull().sum()

for i in table:
    s += i

print(table)
print('Number of empty entries: ', s) if s > 0 else ('No empty entries')

#2

values = []
values_count = []

for i in feature_cols:
    values.append(data[i].unique())
    values_count.append(data[i].value_counts(sort = False))

#numerical values
data.dropna(inplace=True)
data[num_cols[0]] = data[num_cols[0]].astype(int)
for i in num_cols[1:]:
    data[i] = pd.to_numeric(data[i])

f1, (a1, a2, a3) = plt.subplots(3, 1)
f1.set_figheight(22)
f1.set_figwidth(17)

a1.hist(data['ClientPeriod'], bins = (max(data['ClientPeriod']) - min(data['ClientPeriod']))//5)
a1.set_xlabel('Period')
a1.set_ylabel('N of clients')
a1.set_title(feature_cols[0])

a2.hist(data['MonthlySpending'], bins = int(max(data['MonthlySpending']) - min(data['MonthlySpending']))//5)
a2.set_xlabel('Money spent')
a2.set_ylabel('N of clients')
a2.set_title(feature_cols[1])

step = 50
mx = max(data['TotalSpent'])
bins = list(range(0,int(np.ceil(mx/step))*step+step,step))
clusters = pd.cut(data['TotalSpent'],bins,labels=bins[1:])

a3.hist(clusters)
a3.set_xlabel('Money spent')
a3.set_ylabel('N of clients')
a3.set_title(feature_cols[2])

print('ПЕРЕМЕННЫЕ НЕСБАЛАНСИРОВАНЫ')

#categorical values
f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12), (ax13, ax14, ax15, ax16)) = plt.subplots(4, 4)
f.set_figheight(22)
f.set_figwidth(22)

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%({v:d})'.format(p=pct,v=val)
    return my_autopct

for i in range(3, len(values)):
    eval('ax%s.pie(x=values_count[%s], labels = values[%s], autopct = make_autopct(values_count[%s]), pctdistance = 0.6, startangle = 10, textprops={"fontsize" : 10})' % (str(i-2), str(i), str(i), str(i)))
    eval('ax%s.title.set_text("%s")' % (str(i-2), feature_cols[i]))

data = pd.get_dummies(data, columns=cat_cols)
cat_cols_new = []
for col_name in cat_cols:
    cat_cols_new.extend(filter(lambda x: x.startswith(col_name), data.columns))
cat_cols = cat_cols_new


#3
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier




X = data[num_cols+cat_cols]
y = data[target_col]


logReg = LogisticRegression(fit_intercept=True,
                            penalty = 'l1',
                            solver = 'saga',
                            tol=0.001, 
                            max_iter=1000)



cl = make_pipeline(StandardScaler(), logReg)
grid = {'logisticregression__C' : [100, 25, 20, 17, 15, 12, 10, 5]}
gs = GridSearchCV(cl, grid, scoring = 'roc_auc', cv = 10)
gs.fit(X, y)
print(gs.score(X, y))
print(gs.best_score_)
print(gs.best_params_)

#4
for col_name in cat_cols:
    cat_cols_new.extend(filter(lambda x: x.startswith(col_name), test_data.columns))
cat_cols = cat_cols_new

xg = XGBClassifier(objective = 'reg:logistic',
                   learning_rate = 0.1)
params = {'max_depth': [1, 2, 3, 4, 5],
          'booster':['gbtree', 'dart'], 
          'reg_alpha' :[0, 1, 2, 3], 'reg_lambda' : [0, 1, 2, 3]}

gxg = GridSearchCV(xg, params, scoring = 'roc_auc', cv = 10)
gxg.fit(X, y)
print(gxg.best_score_)
print(gxg.best_params_)

best_so_far = gxg.best_params_
best_so_far['objective'] = 'reg:logistic'
best_so_far['learning_rate'] = 0.1

model = XGBClassifier(**best_so_far)
model.fit(X, y)
model.predict_proba(X)

y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)

submission = pd.read_csv('sc.csv')
submission['Churn'] = y_pred
submission.to_csv('msc.csv')



























 










    






