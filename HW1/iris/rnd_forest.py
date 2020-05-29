import numpy as np
import pandas as pd
import random as rnd
from sklearn import tree

def subsample(dataset, target, test, ratio, n_features):
    sample_train = list()
    sample_target = list()

    #random feature selection
    features = [i for i in range(dataset.shape[1])]
    select_feat = rnd.sample(features, n_features)
    select_feat.sort()
    dataset = dataset.iloc[:, select_feat]
	
    #random instance selection
    n_sample = round(len(dataset) * ratio)
    while len(sample_train) < n_sample and len(sample_target) < n_sample:
        assert len(sample_train) == len(sample_target)
        index = rnd.randrange(dataset.shape[0])
        sample_train.append(dataset.iloc[index].values)
        sample_target.append(target.iloc[index].values)
    return sample_train, sample_target, select_feat
		

def bagging_pred(trees, row, select_feats):
    assert len(trees) == len(select_feats)
    row = np.reshape(row,(1,-1))
    predictions = [trees[i].predict(row[:,select_feats[i]]).item() for i in range(len(trees))]
    return max(set(predictions), key=predictions.count)

def randomForest(train, train_target, test, n_trees, n_features, sample_size, criterion='gini', max_depth=None, min_samples_split=2):
    trees = list()
    select_feats = list()
    for i in range(n_trees):
        smp_train, smp_target, select_feat = subsample(train, train_target, test, sample_size, n_features)
        clf = tree.DecisionTreeClassifier(criterion=criterion)
        iris_clf = clf.fit(smp_train, smp_target)
        trees.append(iris_clf)
        select_feats.append(select_feat)
    test = test.values
    preds = [bagging_pred(trees, row, select_feats) for row in test]
    return preds	
