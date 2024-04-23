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

def get_small_data():
    monkey = "C:/Users/random/Documents/KEX_data_stuff/monkeydata.h5ad"
    frontotemporal = "C:/Users/random/Documents/KEX_data_stuff/frontotemporal.h5ad"
    striatum = "C:/Users/random/Documents/KEX_data_stuff/striatum.h5ad"
    anteriorcingulatecortex = "C:/Users/random/Documents/KEX_data_stuff/anteriorcingulatecortex.h5ad"

    m_data = sc.read_h5ad(filename=monkey,backed='r')
    ft_data = sc.read_h5ad(filename=frontotemporal,backed='r')
    str_data = sc.read_h5ad(filename=striatum,backed='r')
    ant_data = sc.read_h5ad(filename=anteriorcingulatecortex,backed='r')

    rng = np.random.default_rng(seed=101)
    m_perm = rng.choice(a=m_data.shape[0], size=min(m_data.shape[0],1000),replace=False)
    ft_perm =  rng.choice(a=ft_data.shape[0], size=min(ft_data.shape[0],1000),replace=False)
    str_perm =  rng.choice(a=str_data.shape[0], size=min(str_data.shape[0],1000),replace=False)
    ant_perm =  rng.choice(a=ant_data.shape[0], size=min(ant_data.shape[0],1000),replace=False)
    
    m_data[m_perm].write("monkey_chunk.h5ad")
    ft_data[ft_perm].write("fronto_temporal_chunk.h5ad")
    str_data[str_perm].write("striatum_chunk.h5ad")
    ant_data[ant_perm].write("anteriorcingulatecortex_chunk.h5ad")


    #np.save('permutated_indices.npy',permutation)


def split_chunk(num,splits=4):
    adata = sc.read_h5ad('data_chunk'+str(num)+'.h5ad')
    rng = np.random.default_rng(seed=53)
    perm = rng.permutation(adata.shape[0])
    split_size = adata.shape[0]//splits
    for i in range(splits):
        ind = perm[i*split_size:(i+1)*split_size]
        temp = adata[ind]
        temp.write('subchunk'+str(num)+'_'+str(i+1)+'.h5ad',compression='gzip')

def split_subchunk(num,k,splits=4):
    adata = sc.read_h5ad('subchunk'+str(num)+'_'+str(k)+'.h5ad')
    rng = np.random.default_rng(seed=53)
    perm = rng.permutation(adata.shape[0])
    split_size = adata.shape[0]//splits
    for i in range(splits):
        ind = perm[i*split_size:(i+1)*split_size]
        temp = adata[ind]
        temp.write('gridchunk'+str(k)+'_'+str(num)+'_'+str(i+1)+'.h5ad')

get_small_data()