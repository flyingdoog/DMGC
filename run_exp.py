import os
import numpy as np
import tensorflow as tf
import time
from inits import *
from utils import get_edges
from DMGC import DMGC
from datasets import load_data
import metrics

tf.random.set_random_seed(1234)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.001, 'initial learning rate.')

# left to default values in main experiments 
flags.DEFINE_integer('epochs', 2000, 'number of epochs to train.')
#logging, saving, validation settings etc.
flags.DEFINE_integer('print_every', 20, "How often to print training info.")
flags.DEFINE_string('dataset', '20news', 'dataset')
flags.DEFINE_integer('emb', 100, "How often to print training info.")
flags.DEFINE_float('beta',2.0, 'weight for beta loss')
flags.DEFINE_boolean('shareW', False, 'share the w')


flags.DEFINE_float('l0', 1.0, 'weight for clustering loss')
flags.DEFINE_float('l1', 1, 'weight for inside-graph regularization term')
flags.DEFINE_float('l2', 0.8, "weight for cross-graph regularization term")
flags.DEFINE_float('l11',1, "weight for second order loss")
flags.DEFINE_float('l01',1, "weight for uniform loss")
flags.DEFINE_integer('centroid_dim', 20, "shared centroid dimmensions")
flags.DEFINE_string('center_method', 'concat', 'center_method')
flags.DEFINE_string('output_dir', './', 'output_dir')
flags.DEFINE_integer('hdim', 128, 'hidden represenation size')



adjs,xs, ys, cnets,Gs= load_data(FLAGS.dataset,norm=True)
n_networks = len(xs)
n_clusters = [len(np.unique(y)) for y in ys]



config = []
for i in range(n_networks):
	config.append([xs[i].shape[-1],FLAGS.hdim,FLAGS.emb])

# preProcess adjs
maskes = [None]*n_networks
beta = FLAGS.beta
for i in range(n_networks):
	maskes[i] = adjs[i].copy()
	maskes[i] *= beta
	maskes[i] += 1

print('network mask done')


cnets_masks = []
for i in range(n_networks):
	cnets_masks.append([None]*n_networks)
	for j in range(n_networks):
		if i == j:
			continue
		tmp = np.sum(cnets[i][j],axis=1) #Ni*1
		tmp[tmp>0]=1
		tmp = np.tile(tmp,(1,n_clusters[i]))
		cnets_masks[i][j]=tmp

print('cross_network mask done')


#print(maskes[0])
dmgc = DMGC(dims=config, n_clusters=n_clusters, init=glorot)

print('build model done')

u_is=[]
u_js=[]
u_labels=[]

for i in range(n_networks):
	G = Gs[i]
	u_i,u_j,u_label = get_edges(G)
	u_is.append(u_i)
	u_js.append(u_j)
	u_labels.append(u_label)


print('prepare to feed data')
dmgc.feedData(adjs,maskes,xs,cnets,cnets_masks,u_js=u_js,u_is=u_is,u_labels=u_labels)

print('feed data done')

begin = time.time()
embs,centers,centroids,re_graphs,best_res,align = dmgc.fit(ys=ys,update_interval=FLAGS.print_every,maxiter=FLAGS.epochs)
end = time.time()
print('total time',(end-begin),'seconds')


nmi_s = []
for i in range(n_networks):
	nmi_s.append(metrics.mask_nmi(ys[i], best_res[i]))
output_dir = FLAGS.output_dir

np.save(output_dir+'/emb',embs)
np.save(output_dir+'/pred',best_res)
np.save(output_dir+'/align',align)
np.save(output_dir+'/centers',centers)
np.save(output_dir+'/centroids',centroids)
