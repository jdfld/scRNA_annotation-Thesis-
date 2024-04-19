from torch.utils.data import Dataset
from torch.utils.data import SequentialSampler
from torch.utils.data import BatchSampler
import scanpy as sc
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
import data_proc_methods as dpm



class GeneDataset(Dataset):

    def __init__(self, file_adress,gene_location='Gene',cell_type_location='supercluster_term',encoder=None,batch_size=2000):
        adata = sc.read_h5ad(filename=file_adress)
        dpm.preprocessing(adata)
        adata = adata[:,adata.var.sort_values(by=gene_location).index]
        if encoder == True:
            encoder = LabelEncoder()
            encoder.fit(adata.obs[cell_type_location])
        if encoder is not None:
            self.labels = torch.LongTensor(encoder.transform(adata.obs[cell_type_location]))
        self.encoder = encoder
        self.genes = adata.var[gene_location]
        self.cells = adata.obs[cell_type_location].cat.categories
        self.features = adata.X
        self.features.data = self.features.data.astype(np.float32)
        self.shape = self.features.shape
        self.batch_size = batch_size
    
    def get_encoder(self):
        return self.encoder

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self,i):
        elem = self.features[i]
        indptr = torch.from_numpy(elem.indptr).type(torch.IntTensor)
        indices = torch.from_numpy(elem.indices).type(torch.IntTensor)
        data = torch.from_numpy(elem.data).type(torch.FloatTensor)
        feature = torch.sparse_csr_tensor(indptr,indices,data,size=elem.shape).to_dense()
        label = self.labels[i]
        device = 'cpu'
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            device = 'mps'
        feature = feature.to(device)
        label = label.to(device)
        return feature,label
    
    def get_batch(self):
        indices = torch.randint(low=0,high=len(self),size=(self.batch_size,))
        return self[indices]
    
    def __iter__(self):
        sizes  = torch.randint(low=0,high=len(self),size=((len(self)+self.batch_size-1)//self.batch_size,self.batch_size))
        for size in sizes:
            yield self[size]

