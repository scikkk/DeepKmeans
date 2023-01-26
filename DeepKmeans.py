'''
Author: scikkk 203536673@qq.com
Date: 2023-01-26 20:56:06
LastEditors: scikkk
LastEditTime: 2023-01-26 22:08:38
Description: file content
'''

import h5py
import torch
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from time import time
from sklearn.preprocessing import LabelEncoder
from VAE import VAE


def DeepKmeans(args):
    # for repeatability
    torch.manual_seed(3407)
    # load data
    data_mat = h5py.File(args.data_file)
    x = np.array(data_mat['X'])
    labels = np.array(data_mat['Y'])
    data_mat.close()

    # preprocess data
    adata = sc.AnnData(x)
    adata.obs['Group'] = labels
    adata.obs['DCA_split'] = 'train'
    adata.obs['DCA_split'] = adata.obs['DCA_split'].astype('category')
    print('Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

    # normalize data
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.filter_cells(adata, min_counts=1)
    adata.raw = adata.copy()
    sc.pp.normalize_per_cell(adata)
    adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)

    # print(adata.X.shape)
    # print(labels.shape)
    model = VAE(input_dim=adata.n_vars, z_dim=16, encodeLayer=[200], decodeLayer=[64,512], sigma=args.sigma, device=args.device)
    # print(str(model))
    t0 = time()
    losses = model.trainAE(X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors, 
                            batch_size=args.batch_size, epochs=args.epoch_num)

    print(f'Training time: {int(time() - t0)}s.')
    # estimate number of clusters by Louvain algorithm on the autoencoder latent representations
    pretrain_latent = model.encodeBatch(torch.tensor(adata.X)).cpu().numpy()
    adata_latent = sc.AnnData(pretrain_latent)
    sc.pp.neighbors(adata_latent, n_neighbors=args.n_neighbors, use_rep="X")
    sc.tl.louvain(adata_latent, resolution=args.resolution)
    y_pred_init = np.asarray(adata_latent.obs['louvain'],dtype=int)
    features = pd.DataFrame(adata_latent.X,index=np.arange(0,adata_latent.n_obs))
    Group = pd.Series(y_pred_init,index=np.arange(0,adata_latent.n_obs),name="Group")
    Mergefeature = pd.concat([features,Group],axis=1)
    cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())
    n_clusters = cluster_centers.shape[0]
    print('Estimated number of cluster centers: ', n_clusters)
    y_pred, _, _ = model.fit(X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors, n_clusters=n_clusters, init_centroid=cluster_centers, 
                y_pred_init=y_pred_init, y=labels)
    print(f'Total time: {int(time() - t0)}s.')
    final_latent = model.encodeBatch(torch.tensor(adata.X)).cpu().numpy()
    np.savetxt(args.final_latent_file, final_latent, delimiter=",")
    np.savetxt(args.predict_label_file, y_pred, delimiter=",", fmt="%i")
    
def plot_res(args):
    data_mat = h5py.File(args.data_file)
    pos = np.array(data_mat['pos'])
    true_label = np.array(data_mat['Y'])
    data_mat.close()
    encoder_x=LabelEncoder()
    true_label=encoder_x.fit_transform(true_label)
    pred_label = np.loadtxt('./pred_labels.txt').astype(int)
    #colors = ['#440453', '#482976', '#3E4A88', '#30688D', '#24828E', '#1B9E8A', '#32B67B', '#6CCC5F', '#B4DD3D', '#FDE73A']
    t_colors = ['#440453', '#482976', '#3E4A88', '#30688D', '#24828E', '#32B67B', 'w', '#6CCC5F']
    p_colors = ['#3E4A88', '#440453', '#482976', '#1B9E8A', '#6CCC5F', '#B4DD3D', '#FDE73A','r','g','b']
    fig = plt.figure(figsize=(11, 5))
    ax1 = fig.add_subplot(1,2,1)
    ax1.scatter(pos[0], pos[1], c=list(map(lambda x: t_colors[x], true_label)),alpha = 1,marker='.',linewidths=2)
    ax1.set_title(f'Sample {args.data_file[-9:-3]}')
    ax2 = fig.add_subplot(1,2,2)
    ax2.scatter(pos[0], pos[1],c=list(map(lambda x: p_colors[x], pred_label)),alpha = 1,marker='.',linewidths=2)
    ax2.set_title('Prediction')
    # plt.savefig(f'pred_{args.data_file[-9:-3]}.png')
    plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_neighbors', type=int, default=76)#76
    parser.add_argument('--resolution', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--data_file', type=str, default='./data/sample_151509.h5')
    parser.add_argument('--epoch_num', type=int, default=24)
    parser.add_argument('--sigma', type=float, default=0.84)
    parser.add_argument('--final_latent_file', type=str, default='final_latent_file.txt', help='Path to store the latent representations.')
    parser.add_argument('--predict_label_file', type=str, default='pred_labels.txt',help='Path to store the clustering results.')
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()
    print(args)
    DeepKmeans(args)
    plot_res(args)

# D:/Users/86176/anaconda3/envs/PyTorch/python.exe DeepKmeans.py --n_neighbors 72 --data_file ./data/sample_151673.h5 --sigma 0.8
# D:/Users/86176/anaconda3/envs/PyTorch/python.exe DeepKmeans.py --n_neighbors 76 --data_file ./data/sample_151509.h5 --sigma 0.84