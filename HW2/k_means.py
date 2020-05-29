import numpy as np
import pandas as pd
from copy import deepcopy

def k_means(data, n_clusters=3):
	#Number of clusters
	k = n_clusters
	#Number of instances
	instances = data.shape[0]
	#Number of features
	feats = data.shape[1]

	#Generate rnd centroids and represent data with mean and std
	mean = np.mean(data, axis = 0)
	std = np.std(data, axis= 0)
	centroids = np.random.randn(k, feats)*std + mean
	
	oldCentroids = np.zeros(centroids.shape)
	newCentroids = deepcopy(centroids)

	#storing the clustering results
	clusters = np.zeros(instances)
	#storing the distances from instances to the centroids
	distances = np.zeros((instances,k))

	# The L2 norm(Euclidean Distance) of new and old centroids
	error = np.linalg.norm(newCentroids - oldCentroids)
	
	#Recursive until the centroids don't move
	while (error>0.00000000000000000000000001):
		# Determine the current cluster
		for i in range(k):
			distances[:, i] =  np.linalg.norm(data - centroids[i], axis=1)
		clusters = np.argmin(distances, axis=1)
		
		#Update the centroids
		oldCentroids = deepcopy(newCentroids)
		for i in range(k):
			newCentroids[i] = np.mean(data[clusters == i], axis = 0)
		error = np.linalg.norm(newCentroids - oldCentroids)

	return clusters,oldCentroids
