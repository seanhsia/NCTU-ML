import numpy as np
import pandas as pd

def kMeansAccuracy(clusters,k):
	clusterCount = clusters.groupby('cluster#').size()
	distri = np.zeros([k,3])

	'''
		distri[i][0]: # of CH in cluster i
		distri[i][1]: # of CU in cluster i
		distri[i][2]: # of FF in cluster i
	
	'''
	for i in range(clusters.shape[0]):
		x = 0
		y = 0
		x = clusters['cluster#'][i]
		if(clusters['pitch_type'][i] == 'CH'):
			y = 0
		elif(clusters['pitch_type'][i] == 'CU'):
			y = 1
		elif(clusters['pitch_type'][i] == 'FF'):
			y = 2
		distri[x][y] += 1
	print(distri)
	for i in range(k):
		for j in range(3):
			distri[i][j] /= sum(distri[i])
	for i in range(k):
		for j in range(3):
			if(distri[i][j] != 0):
				distri[i][j] *= -(np.log2(distri[i][j]))
	print(distri)
	print("Entropy: %f"%sum(sum(distri)))