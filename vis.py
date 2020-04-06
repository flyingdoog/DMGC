from tsne import draw0_with_centers
import numpy as np
from datasets import load_data

output_dir = '.'
embs = np.load(output_dir+'/emb.npy', allow_pickle=True)
best_res = np.load(output_dir+'/pred.npy', allow_pickle=True)
align = np.load(output_dir+'/align.npy', allow_pickle=True)
centers = np.load(output_dir+'/centers.npy', allow_pickle=True)
centroids = np.load(output_dir+'/centroids.npy', allow_pickle=True)

adjs,xs, ys, cnets,Gs= load_data('20news',norm=True)


draw0_with_centers(embs[0],ys[0],centroids[0])

