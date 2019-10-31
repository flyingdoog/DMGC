import numpy as np
import networkx as nx
from numpy.linalg import norm
from scipy.sparse import csr_matrix
from utils import *




def getAdj(file,begin=0,directed=False,weighted=False,maxids=None,addself=False):
	edges=set()
	row = []
	col = []
	data =[]
	rowmax=-1
	colmax =-1
	if weighted:
		leng =3
	else:
		leng = 2
	with open(file) as fin:
		for line in fin:
			ss = line.strip().split()
			if len(ss)<leng:
				continue
			dat = 1
			if weighted:
				dat = float(ss[2])
			rid = int(ss[0])-begin
			cid = int(ss[1])-begin
			if not (rid,cid) in edges:
				row.append(rid)
				col.append(cid)
				data.append(dat)
			if cid>colmax:
				colmax = cid
			if rid > rowmax:
				rowmax = rid

			if not directed:
				if not (cid,rid) in edges:
					row.append(cid)
					col.append(rid)
					data.append(dat)
	rowmax +=1
	colmax +=1
	if not directed:
		if rowmax>colmax:
			colmax=rowmax
		else:
			rowmax=colmax

	if not maxids is None:
		if rowmax < maxids[0]:
			rowmax = maxids[0]
		if colmax < maxids[1]:
			colmax = maxids[1]
	if (not directed) and addself:
		for i in range(rowmax):
			if (i,i) not in edges:
				row.append(i)
				col.append(i)
				data.append(1)

	return csr_matrix((data,(row,col)),shape=(rowmax, colmax)).todense()

def load_20news(layer = 5):
	xs = []
	Gs=[]
	dirPath = './datasets/20news/'
	adjs=[]
	ys=[]
	for i in range(layer):
		filePath = dirPath + 'layers/layer'+str(i)+'.txt'
		A = getAdj(filePath,weighted=True)
		Gs.append(nx.from_numpy_matrix(A))
		X= A
		PT = np.transpose(A/(A.sum(axis=0)))
		tx = np.identity(A.shape[0])

		X = zero_max_normalization(X)
		xs.append(X)
		adjs.append(A)

		yname = dirPath + 'cmtys/cmty'+str(i)+'.txt'

		y = []
		with open(yname) as fin:
			for line in fin:
				label = int(line.strip())
				y.append(label)

		ys.append(np.array(y))


	cnets=[]

	
	for i in range(layer):
		cnets.append([None]*layer)
		for j in range(i):
			cnets[i][j]=np.transpose(cnets[j][i])
		for j in range(i+1,layer):
			filePath = dirPath + 'cross/c'+str(i)+str(j)+'.txt'
			A = getAdj(filePath,weighted=True,directed=True)

			cnets[i][j]=A


	for i in range(layer):
		for j in range(layer):
			if i!=j:
				cnets[i][j]=cnets[i][j]/(cnets[i][j].sum(axis=-1))


	return adjs,xs,ys,cnets,Gs

def getX(A,step=0,alpha=1.0):
	if step==0:
		return A

	PT = np.transpose(A/(A.sum(axis=0)))
	tx = np.identity(A.shape[0])
	for s in range(step):
		tx= alpha*np.matmul(tx,PT)+(1-alpha)*np.identity(A.shape[0])
	X = zero_max_normalization(tx)
	return X

def load_dblp_all(layer = 3):
	adjs =[]
	step = [3,3,3]
	alpha = [0.98,0.98,0.98]

	cnets=[]
	Gs = []
	xs = []
	ys =[None]*layer
	for i in range(layer):
		cnets.append([None]*layer)

	dirPath = './datasets/dblp/'
	begin = 0
	
	filePath = dirPath + 'coauthorNet.txt'
	yname = dirPath + 'alabel.txt'
	
	D = getAdj(filePath,weighted=False,begin=begin,addself=True)
	adjs.append(D)
	# print(G.shape)
	Gs.append(nx.from_numpy_matrix(D))
	xs.append(getX(D,step=step[0],alpha=alpha[0]))
	#print(xs[1])



	ys[0]=[0]*D.shape[0]
	print(D.shape[0])
	with open(yname) as fin:
		aid = 0
		for line in fin:
			label = int(line.strip())
			ys[0][aid]=label
			aid+=1
	
	filePath = dirPath + 'citation.txt'
	yname = dirPath + 'plabel.txt'
	G = getAdj(filePath,weighted=False,begin=begin,addself=True)
	adjs.append(G)
	# print(G.shape)
	Gs.append(nx.from_numpy_matrix(G))
	xs.append(getX(G,step=step[1],alpha=alpha[1]))
	#print(xs[1])



	ys[1]=[1]*G.shape[0]
	print(G.shape[0])
	with open(yname) as fin:
		aid = 0
		for line in fin:
			label = int(line.strip())
			ys[1][aid]=label
			aid+=1

	A_filePath = dirPath + 'APNet.txt'

	A = getAdj(A_filePath,weighted=False,begin=begin,directed=True,maxids=[D.shape[0],G.shape[0]])
	A=A/(A.sum(axis=-1)+1e-5)


	cnets[0][1]=A
	cnets[1][0]=np.transpose(A)

	if layer==3:
		filePath = dirPath + 'co_citation.txt'
		G = getAdj(filePath,weighted=False,begin=begin,addself=True)
		adjs.append(G)
		Gs.append(nx.from_numpy_matrix(G))
		xs.append(getX(G,step=step[2],alpha=alpha[2]))

		ys[2]=ys[1]


	

		cnets[0][2]=A
		cnets[2][0]=np.transpose(A)	


		cnets[0][2]=A
		cnets[2][0]=np.transpose(A)
		cnets[1][2]=np.asmatrix(np.identity(G.shape[0]))

		cnets[2][1]=cnets[1][2]
	





	return adjs,xs,ys,cnets,Gs;

def load_flickr():
	layer = 2
	xs = []
	step = [0,0]
	alpha = [0.98,0.98] #
	dirPath = './datasets/Flickr/layers/'
	yname = './datasets/Flickr/cmty.txt'

	y=[]
	ys = [None]*layer
		
	with open(yname) as fin:
		for line in fin:
			label = int(line.strip())
			y.append(label)

	nodesize = len(y)
	adjs=[]
	Gs=[]
	for i in range(layer):
		filePath = dirPath + 'layer'+str(i)+'.txt'
		A = getAdj(filePath)
		G = nx.from_numpy_matrix(A)
		xfile = dirPath+'x'+str(i)+'_'+str(step[i])+'_'+str(alpha[i])
		Gs.append(G)

		try:
			X = np.load(xfile+'.npy')
		except:
			X = A
			PT = np.transpose(A/(A.sum(axis=0)))
			tx = np.identity(A.shape[0])
			for s in range(step[i]):
				tx= alpha[i]*np.matmul(tx,PT)+(1-alpha[i])*np.identity(A.shape[0])

			if step[i]>0:
				X = tx
			np.save(xfile,X)

		# print(X)
		xs.append(X)
		adjs.append(A)
	
	for i in range(layer):
		ys[i]=np.array(y)

	cnets=[]
	for i in range(layer):
		cnets.append([None]*layer)
		for j in range(layer):
			cnets[i][j]=np.asmatrix(np.identity(nodesize))

	print('read file done')
	return adjs,xs,np.array(ys),cnets,Gs

def load_data(dataset_name,layer=0,norm=True):


	if dataset_name =='dblp':
		return load_dblp_all()

	elif dataset_name =='social' or dataset_name =='flickr':
		return load_flickr()

	elif dataset_name =='20news':
		if layer==0:
			return load_20news()
		else:
			return load_20news(layer)
	else:
		print('Not defined for loading', dataset_name)
		exit(0)






