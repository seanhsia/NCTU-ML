#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[7]:


def kMeansToScatter(clusters):
	p = sns.scatterplot(data=clusters,x = 'x',y = 'y',hue = 'cluster#',
                        style = 'pitch_type',palette = sns.color_palette(n_colors=4),size = 'sizedummy'
                        ,legend = False)
	plt.show()


# In[9]:


def main():
	clusters = pd.read_csv("data_Noah_Cluster.csv",encoding='utf-8')
	print(clusters)
	kMeansToScatter(clusters)
if __name__ == '__main__':
	main()


# In[ ]:




