import scanpy as sc
import anndata as ad
import numpy as np
# A simple script to get split the two large neurons and nonneurons files into 20 smaller chunks for easier management.
# Each chunk consists of the gene expression data of 16219 cells retrieved from 
# The processing of a single chunk takes about 15 minutes
def split_raw_data():
    neurons = sc.read_h5ad(filename='Neurons.h5ad', backed='r')
    non_neurons = sc.read_h5ad(filename='Nonneurons.h5ad',backed='r')
    number_of_cells = neurons.shape[0] + non_neurons.shape[0]

    # Generate random index values. This operation has been precalculated to be able to easily reproduce the results though the code below should generate the same numbers.
    #rng = np.random.default_rng(seed=42)
    #permutation = rng.permutation(number_of_cells)
    #np.save('permutated_indices.npy',permutation)

    permutation = np.load('permutated_indices.npy')

    chunk_sizes = [number_of_cells//20+1]*21
    chunk_sizes[0] = 0
    chunk_sizes[1]-=1

    cur_range = [0,chunk_sizes[1]]
    for i in range(1,2):#,1,len(chunk_sizes)//2+1):
        cur_range[0] = cur_range[1]
        cur_range[1] += chunk_sizes[i]
        ind = permutation[cur_range[0]:cur_range[1]] 
        neuron_chunk = neurons[ind[ind < neurons.shape[0]]]
        non_neurons_chunk = non_neurons[ind[ind >= neurons.shape[0]]-neurons.shape[0]]
        chunk = ad.concat([neuron_chunk,non_neurons_chunk],merge='same')
        chunk.write('data_chunk'+str(i)+'.h5ad',compression='gzip')

def split_chunk(num,splits=4):
    adata = sc.read_h5ad('data_chunk'+str(num)+'.h5ad')
    rng = np.random.default_rng(seed=53)
    perm = rng.permutation(adata.shape[0])
    split_size = adata.shape[0]//splits
    for i in range(splits):
        ind = perm[i*split_size:(i+1)*split_size]
        temp = adata[ind]
        temp.write('subchunk'+str(num)+'_'+str(i+1)+'.h5ad')

