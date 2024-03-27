# Installs:
#!pip install --quiet scvi-tools
#!pip install --quiet scanpy
#!pip install --quiet scikit-misc
#!pip install pynndescent # improves run time of sc.pp.neighbors
#!pip install harmonypy
#!pip install scanorama
#!pip install leidenalg
#!pip install louvain

import scanpy as sc
import numpy as np
#import pandas as pd
#import harmonypy as hrm
#import scvi
#import skmisc
#import scanorama
#import xgboost
#import pynndescent
from anndata.experimental import AnnLoader
import anndata as ad
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from torch import cuda
#def doublet_removal(adata): # remove doublet genes which may cause issues between clusters
#  # fairly poor implementation and very slow. Doublets are few on linarsson data so not necessary
#  scvi.model.SCVI.setup_anndata(adata)
#  vae = scvi.model.SCVI(adata)
#  vae.train()
#  solo = scvi.external.SOLO.from_scvi_model(vae)
#  solo.train()
#  df = solo.predict()
#  df['prediction']= solo.predict(soft= False)
#  df.groupby('prediction').count()
#  doublets = df[df.prediction == 'doublet']
#  adata.obs['doublet'] = adata.obs.index.isin(doublets.index)
#  adata = adata[~adata.obs.doublet]


def load_data(filename):
  adata = sc.read_h5ad(filename=filename)
  preprocessing(adata)
  adata = adata[:,adata.var.sort_values(by='Gene').index]
  encoder = OneHotEncoder(handle_unknown='ignore')
  encoder.fit(adata.obs['supercluster_term'].to_numpy()[:,None])
  return adata,encoder
# loads data and normalizes it
# returns as a Annloader object for usage with torch models
def create_annloader(adata,batch_size=1000):
  return AnnLoader(adata,batch_size=batch_size,shuffle=True,use_cuda=False)



def align(mapping,data):
  # realigns data to fit mapping using matrix multiplication
  # uses efficient sparse matrix mult.
  # requires data sorted according to gene order
  return sparse.csr_matrix.dot(data,mapping)

def map_genes(emb_genes,gene_expr):
  # map genes to the same format as training data as a matrix
  # the mapping is a sparse matrix 
  # that when multiplied with data transforms it into the same shape as the embedders training data
  # assumes data has been sorted along genes
  emb_count = len(emb_genes)
  gene_count = len(gene_expr)
  i,j = 0,0
  mapping = np.zeros((gene_count,emb_count))
  while i < emb_count and j < gene_count:
      if emb_genes[i] == gene_expr[j]:
          mapping[j][i] = 1
          i += 1
          j += 1
      elif emb_genes[i] < gene_expr[j]:
          i += 1
      else:
          j += 1
  return sparse.csr_matrix(mapping)


def preprocessing(adata,min_genes=-1,min_cells=-1,filter_highly_variable=False):
  if min_genes != -1: # remove cells with fewer than min_genes counts
    sc.pp.filter_cells(adata,min_genes=min_genes)
  if min_cells != -1:
    sc.pp.filter_genes(adata,min_cells=min_cells)
  sc.pp.normalize_total(adata, target_sum=1e4) #normalize every cell to 10,000 UMI
  sc.pp.log1p(adata)
  if filter_highly_variable: # extracts the 2000 genes with most variance
    sc.pp.highly_variable_genes(adata,n_top_genes=2000, subset = True, flavor = 'seurat')

def pca(adata,plot=False):
  sc.tl.pca(adata,svd_solver = 'arpack') # performs pca for adata with default arnoldi package for solving large scale eigenvalue matrices.
  if plot:
    sc.pl.pca_variance_ratio(adata, log=True, n_pcs = 50)

def integration(adata,method='harmony',key='donor_id'):
  adata = adata[adata.obs.sort_values(by=key).index]
  if method == 'harmony':
    harmonized = hrm.run_harmony(adata.obsm['X_pca'], adata.obs, ['donor_id'])# Significant cost, ca 2 min using 50 pca, watch out for ram issues
    adata.obsm['X_harmony'] = harmonized.Z_corr.T
  elif method == 'scanorama':
    sc.external.pp.scanorama_integrate(adata,key=key)
  return adata

def neighbor_umap(adata,use_rep,plot=False,color=None):
  sc.pp.neighbors(adata,n_pcs=min(128,adata.obsm[use_rep].shape[1]),use_rep=use_rep)
  # Create Umap
  sc.tl.umap(adata)
  if plot:# Plot Umap
    sc.pl.umap(adata,color=color)

def pipeline(file_adress):
  adata = sc.read_h5ad(file_adress)
  preprocessing(adata,min_genes=500,min_cells=10,filter_highly_variable=True)
  pca(adata)
  #adata = integration(adata)
  return adata

def community(adata,method='leiden',resolution=1):
  if method == 'leiden':
    sc.tl.leiden(adata,resolution=resolution)
  elif method =='louvain':
    sc.tl.louvain(adata,resolution=resolution)