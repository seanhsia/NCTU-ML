import numpy as np
import pandas as pd
from rnd_forest import *
from sklearn.utils import shuffle
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import *
from sklearn.tree import export_graphviz

iris = pd.read_csv('iris.csv')
feature_names = list(iris)
feature_names.remove('target_names')
del feature_names[0]
x = iris[feature_names]
y = iris.target_names

data = pd.concat([x,y], axis=1)
data = shuffle(data)

'''
# for rnd_frst KFold
data1 = data.iloc[0:50, 0:4]
data1_target = data.iloc[0:50,[4]]
data2 = data.iloc[50:100, 0:4]
data2_target = data.iloc[50:100, [4]]
data3 = data.iloc[100:150, 0:4]
data3_target = data.iloc[100:150, [4]]

data12 = pd.concat([data1,data2], axis=0)
data12_target = pd.concat([data1_target, data2_target], axis=0)
data13 = pd.concat([data1,data3], axis=0)
data13_target = pd.concat([data1_target, data3_target], axis=0)
data23 = pd.concat([data3,data2], axis=0)
data23_target = pd.concat([data3_target, data2_target], axis=0)

'''
#settings
criterion = 'gini'
n_tests = 1
n_splits = 3
avg_acc_dt = 0
avg_acc_rf = 0
avg_cf_matrix_dt = np.zeros(shape=(3,3))
avg_cf_matrix_rf = np.zeros(shape=(3,3))

#model
for test in range(n_tests):
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
    train_y = train_y.to_frame()
    test_y = test_y.to_frame()
    #decision tree model
    classifier = tree.DecisionTreeClassifier(criterion=criterion)
    iris_classifier = classifier.fit(train_x, train_y)

    '''
    #output the figure
    dot_data = tree.export_graphviz(iris_classifier, out_file="iris_tree_entropy.dot", 
        	                 feature_names=iris.feature_names,  
        	                 class_names=iris.target_names,  
        	                 filled=True, rounded=True,  
        	                 special_characters=True)  
    '''
    test_y_pred = iris_classifier.predict(test_x)
    #print(iris.target_names)
    cf_matrix_dt = confusion_matrix(test_y, test_y_pred)
    avg_cf_matrix_dt += cf_matrix_dt
    #print("\nconfusion matrix of decision tree model")
    #print(cf_matrix_dt)
    acc = accuracy_score(test_y, test_y_pred)
    avg_acc_dt += acc
    #print("decision tree acc: %f" %acc)


    #random forest
    #settings
    n_trees = 10
    #for iris dataset
    n_features = 4
    sample_size = 0.7


    test_y_pred = randomForest(train_x, train_y, test_x, n_trees, n_features, sample_size, criterion)

    cf_matrix_rf = confusion_matrix(test_y, test_y_pred)
    avg_cf_matrix_rf += cf_matrix_rf
    #print("\nconfusion matrix of random forest model")
    #print(cf_matrix_rf)   
    acc = accuracy_score(test_y, test_y_pred)
    avg_acc_rf += acc

avg_acc_dt /= n_tests
avg_acc_rf /= n_tests
avg_cf_matrix_dt /= n_tests
avg_cf_matrix_rf /= n_tests

print("\naverage confusion matrix of decision tree model")
print(avg_cf_matrix_dt)
print("decision tree average acc: %f" %avg_acc_dt)
print("\naverage confusion matrix of random forest model")
print(avg_cf_matrix_rf)
print("random forest average acc: %f" %avg_acc_rf)
