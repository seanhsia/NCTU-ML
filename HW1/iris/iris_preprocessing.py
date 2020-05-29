import pandas as pd
import numpy as np
import sys

iris = list()
with open('iris.data.txt', 'r')as f:
    for line in f:
        iris.append(list(line.split(',')))
iris = pd.DataFrame(iris, columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'target_names'])

#not in dataset
iris = iris.drop([150])

#tokenize
iris_type = iris.groupby('target_names')
iris_type = list(iris_type.size().index)
tokenize_target = {num:token for token, num in enumerate(iris_type)}
iris['target_names'] = iris['target_names'].map(tokenize_target)

iris_data = iris.drop(columns=['target_names'])
iris_data = iris_data.astype(float)
iris_target = iris.target_names.to_frame()
print(type(iris_target))
iris = pd.concat([iris_data, iris_target], axis=1)

iris.to_csv('iris.csv')
#stat_data.to_csv('iris_statistic_features.csv')


