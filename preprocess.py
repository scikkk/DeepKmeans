'''
Author: scikkk 203536673@qq.com
Date: 2022-06-25 12:56:02
LastEditors: scikkk
LastEditTime: 2023-01-25 01:44:33
Description: file content
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import scipy as sp
import pandas as pd
import scanpy as sc




def read_dataset(adata):
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error

    adata.obs['DCA_split'] = 'train'
    adata.obs['DCA_split'] = adata.obs['DCA_split'].astype('category')
    print('### Autoencoder: Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))
    return adata


def normalize(adata):
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.filter_cells(adata, min_counts=1)
    adata.raw = adata.copy()
    sc.pp.normalize_per_cell(adata)
    adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    return adata
