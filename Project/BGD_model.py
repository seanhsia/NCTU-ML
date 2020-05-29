import numpy as np
import pandas as pd
from sklearn import tree, ensemble
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
data = pd.read_csv('BanGDream_tokenize.csv')
data_song = pd.read_csv('BanGDream_song_tokenize.csv')

data = shuffle(data)
data_song = shuffle(data_song)

X = data.drop(["Unnamed: 0","Band"],axis=1)
y = data["Band"]
X_song = data_song.drop(["Unnamed: 0","Band"],axis=1)
y_song = data_song["Band"]



def train(X, y, model, criterion):
    if model == "KNN" or "SVM":
        pca = PCA(n_components=3)
        X = pca.fit_transform(X)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
    if model == "Tree":
        classifier = tree.DecisionTreeClassifier(criterion=criterion)
        band_clf = classifier.fit(train_x, train_y)
        
        '''
        #output the figure
        dot_data = tree.export_graphviz(band_clf, out_file="band_tree_"+criterion+".dot",
                                feature_names = list(X),
                                class_names = ["A", "HHW","P*P", "PPP", "R" ],
                                filled = True, rounded=True,
                                special_characters=True)
        '''
        test_y_pred = band_clf.predict(test_x)

        print("A, HHW, P*P, PPP, R")
        print(confusion_matrix(test_y, test_y_pred))
        acc = accuracy_score(test_y, test_y_pred)
        print("decision tree acc: %f" %acc)

        frst_clf = ensemble.RandomForestClassifier(n_estimators=10, criterion=criterion)
        band_frst = frst_clf.fit(train_x, train_y)
        test_y_pred = band_frst.predict(test_x)

        print("A, HHW, P*P, PPP, R")
        print(confusion_matrix(test_y, test_y_pred))
        acc = accuracy_score(test_y, test_y_pred)
        print("random forest acc: %f" %acc)

    elif model == "KNN":
        KNN = KNeighborsClassifier(n_neighbors=criterion)
        band_KNN = KNN.fit(train_x, train_y)
        
        test_y_pred = band_KNN.predict(test_x)

        print("A, HHW, P*P, PPP, R")
        print(confusion_matrix(test_y, test_y_pred))
        acc = accuracy_score(test_y, test_y_pred)
        print("KNN acc: %f" %acc)

    elif model == "SVM":
        svm = SVC(kernel = criterion, probability=True)
        band_svm = svm.fit(train_x, train_y)

        test_y_pred = band_svm.predict(test_x)
        print("A, HHW, P*P, PPP, R")
        print(confusion_matrix(test_y, test_y_pred))
        acc = accuracy_score(test_y, test_y_pred)
        print("SVM acc: %f" %acc)
       

print("-----------------------IN IMFORMATION BASED-------------------------")
model = "Tree"
criterion = "gini" 
train(X,y,model, criterion)
print("----------------------IN KNN MODEL----------------------------------")
model = "KNN"
criterion = 3 
train(X,y,model, criterion)
print("---------------------IN SVM MODEL-----------------------------------")
model = "SVM"
criterion = "poly" 
train(X,y,model, criterion)
print("-----------------------IN IMFORMATION BASED-------------------------")
model = "Tree"
criterion = "gini" 
train(X_song,y_song,model, criterion)
print("----------------------IN KNN MODEL----------------------------------")
model = "KNN"
criterion = 3 
train(X_song,y_song,model, criterion)
print("---------------------IN SVM MODEL-----------------------------------")
model = "SVM"
criterion = "poly" 
train(X_song,y_song,model, criterion)
