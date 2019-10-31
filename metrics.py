import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

nmi = normalized_mutual_info_score
ari = adjusted_rand_score


def eval_acc(tru, pre):
		# true label: numpy, vector in col
		# pred lable: numpy, vector in row
	tru = np.array(tru)
	num_labels = tru.shape[0]
	# accuracy
	tru = np.reshape(tru, (num_labels))
	#set_tru = set(tru.tolist())
	set_pre = set(pre.tolist())
	#nclu_tru = len(set_tru) # in case that nclu_tru != the preset cluster num
	nclu_pre = len(set_pre)
	
	correct = 0
	for i in range(nclu_pre):
		flag = list(set_pre)[i]
		idx = np.argwhere(pre == flag)
		correct += max(np.bincount(tru[idx].T[0].astype(np.int64)))
	acc = correct / num_labels

	return acc


def mask_nmi(y_true, y_pred):

	n_y_ture = []
	n_y_pred = []

	for i in range(len(y_true)):
		if y_true[i]>=0:
			n_y_ture.append(y_true[i])
			n_y_pred.append(y_pred[i])
	return nmi(n_y_ture,n_y_pred)
