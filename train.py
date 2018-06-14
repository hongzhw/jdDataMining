import sys
reload(sys)
sys.setdefaultencoding('gbk')
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('train_featureV1.csv')
test = pd.read_csv('test_featureV1.csv')

dtrain = lgb.Dataset(train.drop(['uid', 'label'], axis=1),label=train.label)
dtest = lgb.Dataset(test.drop(['uid'], axis=1))

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'min_data_in_leaf': 10,
    'num_leaves': 80,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'verbosity': -1,
}


def evalMetric(preds, dtrain):
    label = dtrain.get_label()

    pre = pd.DataFrame({'preds': preds, 'label': label})
    pre = pre.sort_values(by='preds', ascending=False)

    auc = metrics.roc_auc_score(pre.label, pre.preds)

    pre.preds = pre.preds.map(lambda x: 1 if x >= 0.5 else 0)

    f1 = metrics.f1_score(pre.label, pre.preds)

    res = 0.6 * auc + 0.4 * f1

    return 'res', res, True


lgb.cv(lgb_params, dtrain, feval=evalMetric, early_stopping_rounds=100, verbose_eval=5, num_boost_round=10000, nfold=3, metrics=['evalMetric'])
model = lgb.train(lgb_params, dtrain, feval=evalMetric, verbose_eval=5, num_boost_round=300, valid_sets=[dtrain])
pred = model.predict(test.drop(['uid'], axis=1))

res = pd.DataFrame({'uid': test.uid, 'label': pred})
res = res.sort_values(by='label', ascending=False)
res.label = res.label.map(lambda x: 1 if x >= 0.5 else 0)
res.label = res.label.map(lambda x: int(x))
res.to_csv('lgb-baseline_61.csv', index=False, header=False, sep=',', columns=['uid', 'label'])

