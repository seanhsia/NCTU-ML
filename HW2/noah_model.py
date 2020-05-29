import numpy as np
import pandas as pd
from k_means import *
from kMeansAccuracy import *

def main():

	# # of clusters
	k = 3
	
	data = pd.read_csv("data_Noah_Preprocessing.csv", encoding='utf-8')
	data = data.drop(['pitch_type'],axis=1)
	result,finalCentroids = k_means(data.values,n_clusters=k)
	#fill the dataframe with kMeans result
	cluster = pd.DataFrame(data=result,columns=['cluster#'])
	outData = pd.read_csv("data_Noah_Preprocessing.csv", encoding='utf-8')
	outData = appendCol(outData,cluster)
	outData = outData.sort_values(by = ['cluster#','pitch_type'])
	kMeansAccuracy(outData,k)

	centroids = pd.DataFrame(data = finalCentroids,columns = ['x','y','speed','az'])
	#dummy values for visualization
	dummy = pd.DataFrame(data = {'pitch_type':['GG']*k,'cluster#':[k]*k})
	centroids = appendCol(centroids,dummy)
	#dummy values for visualization
	dummy2 = pd.DataFrame(data = {'sizedummy':['4']*data.shape[0]+['10']*k})
	outData = outData.append(centroids,sort = False,ignore_index = True)
	outData = appendCol(outData,dummy2)
	
	outData.to_csv("data_Noah_Cluster.csv",index_label=False,index=False)
	
def appendCol(data,toAppend):
	d = data.T
	toA = toAppend.T
	d = d.append(toA,sort = False)
	d = d.T
	return d
	
	
	

if __name__ == '__main__':
	main()