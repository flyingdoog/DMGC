from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from time import time
import numpy as np
import tensorflow as tf
import metrics
from layers import *
from inits import *
from utils import *

flags = tf.app.flags
FLAGS = flags.FLAGS

sess = tf.Session()

class GraphAutoEncoder():

	def __init__(self,nid,dims, act=tf.nn.relu, init=glorot, x=None,mask=None):

		self.input = x
		n_stacks = len(dims) - 1
		# input
		h = x
		self.mask = mask
		# internal layers in encoder
		for i in range(n_stacks-1):
			h = Dense(output_dim=dims[i + 1],input_dim = dims[i],activation=act, kernel_initializer=init, name='encoder_%d_%d' % (nid,i))(h)

		# hidden layer
		h = Dense(output_dim=dims[-1],input_dim = dims[-2], kernel_initializer=init, name='encoder_%d_%d' % (nid,n_stacks - 1))(h) # hidden layer, features are extracted from here

		y = h
		# internal layers in decoder
		for i in range(n_stacks-1, 0, -1):
			y = Dense(output_dim=dims[i],input_dim=dims[i+1], activation=act, kernel_initializer=init, name='decoder_%d_%d' % (nid,i))(y)

		# output
		y = Dense(output_dim=dims[0],input_dim=dims[1], name='decoder_%d_0'% nid)(y)

		self.emb=h
		self.output=y
		diff = (self.output - self.input)*self.mask
		self.pretrain_loss = tf.reduce_mean(tf.pow(diff, 2))
		self.pretrain = tf.train.AdamOptimizer(1e-3).minimize(self.pretrain_loss)	


class ClusteringLayer():

	def __init__(self, network_id,center_dim, output_dim, n_clusters, Cs=None,centers=None, method='concat',w=None,**kwargs):

		self.n_clusters = n_clusters
		self.output_dim = output_dim
		n_networks = len(centers)
		reshaped_centers = []
		self.network_id= network_id


		if method=='concat':
			for i in range(n_networks):
				if i==self.network_id:
					reshaped_centers.append(centers[i])
					network_id = i
				else:
					reshaped_centers.append(tf.matmul(Cs[i],centers[i]))

			concat_centers = tf.concat(reshaped_centers,axis=-1)
			if w==None:			
				self.w  = glorot([n_networks*center_dim,output_dim],name='centeroid_map_'+str(network_id))
			else:
				self.w = w
			self.centroids =tf.nn.relu(tf.matmul(concat_centers,self.w))

		elif method == 'ave':
			for i in range(n_networks):
				if i==self.network_id:
					continue
				else:
					reshaped_centers.append(tf.matmul(Cs[i],centers[i]))

			ave_centers = tf.add_n(reshaped_centers)/(n_networks-1)
			concat_centers = tf.concat([centers[network_id],ave_centers],axis=-1)
			if w==None:
				self.w  = glorot([2*center_dim,output_dim],name='centeroid_map_'+str(network_id))
			else:
				self.w = w
			self.b  = zeros([output_dim], name='bias'+str(network_id))
			self.centroids =tf.nn.relu(tf.matmul(concat_centers,self.w+self.b))
		else:
			self.centroids = centers[network_id]



	def call(self, inputs, **kwargs):
		""" student t-distribution, as same as used in t-SNE algorithm.
				q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
		Arguments:
			inputs: the variable containing data, shape=(n_samples, n_features)
		Return:
			q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
		"""

		q = 1.0 / (1.0 + (tf.reduce_sum(tf.square(tf.expand_dims(inputs, axis=1) - self.centroids), axis=2)))
		q = tf.transpose(tf.transpose(q) / tf.reduce_sum(q, axis=1))


		return q

def kl_divergence(p, q): 
	return tf.reduce_sum(p * tf.log(p/q))

class DMGC(object):
	def __init__(self,
				dims,
				n_clusters=[10],
				init=glorot,
				):

		super(DMGC, self).__init__()
		n_networks = len(dims)

		self.input_dims = []
		self.n_networks = n_networks
		self.inputs=[]
		self.masks = []
		self.adjs=[]
		self.center_dim = FLAGS.centroid_dim

		l0 =  FLAGS.l0
		l1 =  FLAGS.l1
		l2 =  FLAGS.l2


		l01 = FLAGS.l01 #uniform
		l11 = FLAGS.l11 #second


		learning_rate = FLAGS.learning_rate
		shareW = None
		if FLAGS.shareW:
			shareW = glorot([n_networks*self.center_dim,dims[0][-1]],name='centeroid_map_')



		self.bias_mats=[]
		for i in range(n_networks):
			self.input_dims.append(dims[i][0])
			self.inputs.append(tf.placeholder(name='inputs'+str(i),dtype=tf.float32, shape=[None,self.input_dims[i]]))
			self.masks.append(tf.placeholder(name='mask'+str(i),dtype=tf.float32, shape=[None,self.input_dims[i]]))
			self.adjs.append(tf.placeholder(name='adjs'+str(i),dtype=tf.float32, shape=[None,None]))


		self.dims = dims#[][]
		
		#stacked autoencoder configs
		self.n_stacks =[]
		for i in range(n_networks):
			self.n_stacks.append(len(self.dims[i]) - 1)

		self.n_clusters = n_clusters

		self.aes = []
		self.embs = []
		self.encoder_outputs=[]

		for i in range(n_networks):
			self.aes.append(GraphAutoEncoder(nid = i,dims=self.dims[i], init=init,x=self.inputs[i],mask=self.masks[i]))
			self.embs.append(self.aes[i].emb)
			self.encoder_outputs.append(self.aes[i].output)

		self.clayers=[None]*n_networks
		self.qs=[None]*n_networks

		#clustering loss of each network
		self.losses=[None]*n_networks
		self.uniform_losses=[None]*n_networks
		self.uniforms=[None]*n_networks
		self.centers = [None]*n_networks
		self.centroids=[None]*n_networks
		for i in range(n_networks):
			self.centers[i]=glorot([n_clusters[i],self.center_dim],name='centers_'+str(i))



		self.Cs=[None]*n_networks

		for i in range(n_networks):
			self.Cs[i]=[None]*n_networks
			self.Cs[i][i]=tf.constant(0.0)
			mu_i = self.centers[i] #ki * d
			for j in range(n_networks):
				mu_j = self.centers[j] #ki * d
				if i==j:
					continue
				multiply_i = tf.constant([self.n_clusters[j],1])
				temp = tf.tile(mu_i,multiply_i)
				tile_vec_i = tf.reshape(temp, [multiply_i[0], tf.shape(mu_i)[0],tf.shape(mu_i)[1]]) #kj *ki *d
				re_tile_vec_i = tf.transpose(tile_vec_i,[1,0,2]) #ki *kj *d

				multiply_j = tf.constant([self.n_clusters[i],1])
				temp = tf.tile(mu_j,multiply_j)
				tile_vec_j = tf.reshape(temp, [multiply_j[0], tf.shape(mu_j)[0],tf.shape(mu_j)[1]]) #ki *kj *d
				
				diff = re_tile_vec_i-tile_vec_j
				reduce_sum_diff = tf.reduce_sum(tf.pow(diff,2),axis=2)


				t_sim = 1.0 / (1+reduce_sum_diff)
				logtis = t_sim
				self.Cs[i][j] = logtis / (tf.reduce_sum(logtis, axis=0))



		for i in range(n_networks):
			if n_clusters[i] > 0:
				self.clayers[i]=ClusteringLayer(network_id = i,center_dim = self.center_dim,output_dim=dims[i][-1],n_clusters=self.n_clusters[i],name='clustering'+str(i),Cs=self.Cs[i],centers=self.centers,w = shareW, method = FLAGS.center_method)
				self.qs[i] = self.clayers[i].call(self.embs[i])
				qave=tf.reduce_mean(self.qs[i],axis=0)
				self.uniforms[i]=tf.constant(1.0/n_clusters[i], shape=[n_clusters[i]])
				self.uniform_losses[i]=kl_divergence(qave,self.uniforms[i])
				self.centroids[i]=self.clayers[i].centroids




		self.closses=[]


		for i in range(n_networks):
			cluster_loss = tf.reduce_sum(self.qs[i] * self.qs[i], axis=1)
			self.closses.append(-tf.reduce_mean(tf.log_sigmoid(cluster_loss)))
		self.cluster_loss = tf.add_n(self.closses)

		# first order contraints on embs
		self.u_is=[]
		self.u_js=[]
		self.u_i_qs=[]
		self.u_j_qs=[]
		self.u_labels=[]
		self.first_order_on_q=[]

		for i in range(n_networks):
			self.u_is.append(tf.placeholder(name='u_i'+str(i),dtype=tf.int32,shape=[None]))
			self.u_js.append(tf.placeholder(name='u_j'+str(i),dtype=tf.int32,shape=[None]))
			self.u_labels.append(tf.placeholder(name='u_labels'+str(i),dtype=tf.float32,shape=[None]))
			self.u_i_qs.append(tf.gather(self.qs[i],self.u_is[i]))
			self.u_j_qs.append(tf.gather(self.qs[i],self.u_js[i]))
			inner_product_first_order_q = tf.reduce_sum(self.u_i_qs[i] * self.u_j_qs[i], axis=1)
			self.first_order_on_q.append(-tf.reduce_mean(tf.log_sigmoid(self.u_labels[i] *inner_product_first_order_q)))
			
		self.first_order_loss= tf.add_n(self.first_order_on_q)



		self.cross_networks =[]
		self.cross_networks_masks=[]
		for i in range(n_networks):
			cneti=[None]*self.n_networks
			self.cross_networks.append(cneti)
			self.cross_networks_masks.append([None]*self.n_networks)
			for j in range(n_networks):
				if(i==j):
					continue
				self.cross_networks[i][j]=tf.placeholder(name='cnet'+str(i)+'_'+str(j),dtype=tf.float32, shape=[None,None])
				self.cross_networks_masks[i][j]=tf.placeholder(name='cnetmask'+str(i)+'_'+str(j),dtype=tf.float32, shape=[None,None])



		#sum of pretrain loss of each network
		self.second_order_loss = self.aes[0].pretrain_loss
		for i in range(1,n_networks):
			self.second_order_loss += self.aes[i].pretrain_loss



		
		self.Alosses = []
		for i in range(n_networks):
			for j in range(n_networks):
				if i==j:
					continue
				tqj =tf.transpose(self.qs[j])
				tqi = tf.transpose(self.qs[i])

				Cij_tqj=tf.matmul(self.Cs[i][j],tqj)#ki X kj  kj X Nj = ki X Nj


				qi_Cij_tqj=tf.matmul(self.qs[i],Cij_tqj)

				# try to add Aij
				CA = tf.matmul(self.cross_networks[i][j],self.inputs[j])
				ACA = tf.matmul(self.inputs[i],CA)
				diff = self.cross_networks[i][j] -qi_Cij_tqj

				cij_tqj_t = tf.transpose(Cij_tqj)#Nj * Ki				
				sij_cij_tqj_t= tf.matmul(self.cross_networks[i][j],cij_tqj_t)#Ni*ki


				diff = self.qs[i]-sij_cij_tqj_t
				mask_diff = tf.multiply(self.cross_networks_masks[i][j],diff)
				self.Alosses.append(tf.reduce_mean(tf.pow(mask_diff, 2)))

	

		if len(self.Alosses)>0:
			self.cross_loss = tf.add_n(self.Alosses)
		else:
			self.cross_loss = tf.constant(0.0)

		self.uniloss = tf.add_n(self.uniform_losses)

		self.cluster_loss += l01*self.uniloss
		self.inside_loss = self.first_order_loss +l11*self.second_order_loss
		self.all_loss =l0*self.cluster_loss+l1*self.inside_loss+l2*self.cross_loss


		self.jointTrain=  tf.train.AdamOptimizer(learning_rate).minimize(self.all_loss)

		self.feed_dict_val={}
		sess.run(tf.global_variables_initializer())



	def feedData(self, adjs,masks,xs,cnets,cnets_masks,u_is=None,u_js=None,u_labels=None):

		for i in range(self.n_networks):
			self.feed_dict_val.update({self.inputs[i]: xs[i]})
			self.feed_dict_val.update({self.masks[i]: masks[i]})
			self.feed_dict_val.update({self.adjs[i]: adjs[i]})

			if u_is!=None and u_js!=None:
				self.feed_dict_val.update({self.u_is[i]:u_is[i]})
				self.feed_dict_val.update({self.u_js[i]:u_js[i]})
				self.feed_dict_val.update({self.u_labels[i]:u_labels[i]})


		for i in range(self.n_networks):
			for j in range(self.n_networks):
				if i!=j:
					self.feed_dict_val.update({self.cross_networks[i][j]:cnets[i][j]})
					self.feed_dict_val.update({self.cross_networks_masks[i][j]:cnets_masks[i][j]})


	def fit(self,ys=None, maxiter=2000,print_every=140,test_time = False):

		embs=[None]*self.n_networks

		second_order_loss = 0
		index = 0
		uniloss = 0
		cluster_loss = 0
		ps = [None]*self.n_networks
		qs=[None]*self.n_networks
		maxnmi =[0]*self.n_networks
		floss =0
		centroids=[None]*self.n_networks
		centers=[None]*self.n_networks
		nmis = [0]*self.n_networks
		cross_loss = 0
		loss = 0
		best_re_graphs = [None]*self.n_networks
		re_graphs = [None]*self.n_networks

		Cs=[None]*self.n_networks
		atten_2s=[None]*self.n_networks
		best_embs=[None]*self.n_networks
		best_centroids=[None]*self.n_networks
		res = [None]*self.n_networks
		best_centers = [None]*self.n_networks

		maxnmi12 = 0 

		for i in range(self.n_networks):
			Cs[i]=[None]*self.n_networks
		clfs = []
		for i in range(self.n_networks):
			clfs.append(KMeans(n_clusters=self.n_clusters[i], random_state=0))

		for ite in range(int(maxiter)):
			if (not test_time) and ite % print_every == 0:
				print('----------------------epoch %d----------------------' % (ite))
				print('loss= %.5f' % (loss))

				for i in range(self.n_networks):
					if self.n_clusters[i]>0:
						q = sess.run(self.qs[i], feed_dict=self.feed_dict_val)

						
						y_pred = q.argmax(1)
						if ys[i] is not None:
							nmi_s = np.round(metrics.mask_nmi(ys[i], y_pred), 5)
							nmis[i] = nmi_s
							res[i] = y_pred
							# print('epoch %d, network %d:, maxnmi%.5f, nmi = %.5f' % (ite,i,maxnmi[i], nmi_s),'; loss',loss,' ; 2ndloss',second_order_loss,'closs=', cluster_loss,' ; floss=', floss ,' cross_loss ',cross_loss)
							print('network: %d:, maxnmi:%.5f, nmi = %.5f' % (i,maxnmi[i], nmi_s))
							if maxnmi[i] < nmi_s:
								maxnmi[i] = nmi_s

				if maxnmi12 <(nmis[0]+nmis[1]):
					maxnmi12 = (nmis[0]+nmis[1])
					best_embs = embs
					best_centroids = centroids
					best_centers = centers
					best_re_graphs = re_graphs

			_,centers,centroids,second_order_loss,Cs,cluster_loss,floss,cross_loss,embs,loss,re_graphs = sess.run([self.jointTrain,self.centers,self.centroids,self.second_order_loss, self.Cs,self.cluster_loss,self.first_order_loss,self.cross_loss,self.embs,self.all_loss,self.encoder_outputs], feed_dict=self.feed_dict_val)

		return embs,centers,centroids,re_graphs,res,Cs