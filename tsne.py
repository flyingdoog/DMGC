# coding='utf-8'

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE

def get_data(db):
	data = np.load(db)
	print(data.shape)
	n_samples, n_features = data.shape
	return data, n_samples, n_features

def draw0_with_centers(data, labels, center):
	labels = labels -1
	tsne = TSNE(n_components=2, init='pca', random_state=0)
	n_instance_0 = data.shape[0]

	data = np.concatenate((data,center),axis=0)   


	x_min, x_max = np.min(data, 0), np.max(data, 0)
	data = (data - x_min) / (x_max - x_min)

	x_tsne = tsne.fit_transform(data)

	result_0 = x_tsne[:n_instance_0,:]
	center_ = x_tsne[n_instance_0:,:]

	n_clusters_0 = len(np.unique(labels))

	colors_0 = ['gold','firebrick','red','slategray','darksalmon','mediumblue','maroon','brown','deeppink','paleturquoise','darkslategray','mediumseagreen','lime','orchid']


	for label in range(n_clusters_0):
		# print(label)
		subset = []
		for i in range(n_instance_0):
			if labels[i]==label:
				subset.append(result_0[i])
		if len(subset)>0:
			subset = np.array(subset)
			plt.scatter(subset[:,0],subset[:,1],color=colors_0[label])


	for label in range(n_clusters_0):
		for i in range(n_instance_0):
			plt.scatter(center_[label][0],center_[label][1],color='black',s=100)

	plt.show()	

